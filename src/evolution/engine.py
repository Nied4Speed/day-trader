"""Evolutionary engine: elimination, crossover, mutation, and spawning.

Runs as a batch job after market close. Reads session performance data,
eliminates the worst models, merges survivors' parameters into hybrid
offspring, applies bounded mutations, and writes the next generation
to the database.
"""

import copy
import logging
import random
from datetime import datetime
from typing import Optional

import numpy as np

from src.core.config import Config
from src.core.database import (
    GenerationRecord,
    ModelStatus,
    PerformanceSnapshot,
    TradingModel,
    get_session,
)
from src.core.fitness import FitnessScore, compute_fitness
from src.core.performance import PerformanceTracker
from src.strategies.registry import create_strategy, get_strategy_types

logger = logging.getLogger(__name__)


class EvolutionEngine:
    """Handles the evolutionary cycle between trading sessions."""

    def __init__(self, config: Config):
        self.config = config

    def run_evolution(
        self,
        session_date: str,
        tracker: PerformanceTracker,
    ) -> dict:
        """Execute the full evolutionary cycle.

        1. Rank models by fitness
        2. Eliminate bottom performers
        3. Crossover survivors to produce offspring
        4. Mutate offspring parameters
        5. Spawn new generation
        6. Log generational record

        Returns a summary of what happened.
        """
        leaderboard = tracker.get_leaderboard()
        if not leaderboard:
            logger.warning("No models to evolve")
            return {"error": "no_models"}

        # Separate protected models (COLLAB, CFA-generated) from competitive pool
        protected_types = {"collab", "cfa_generated"}
        protected = [m for m in leaderboard if m.strategy_type in protected_types]
        competitive = [m for m in leaderboard if m.strategy_type not in protected_types]

        if not competitive:
            logger.warning("No competitive models to evolve")
            return {"error": "no_competitive_models"}

        total = len(competitive)
        elimination_count = min(
            self.config.arena.weekly_elimination_count,
            max(1, total - 1),  # keep at least 1 survivor
        )

        # Split competitive models into survivors and eliminated
        survivors = competitive[:-elimination_count]
        eliminated = competitive[-elimination_count:]

        logger.info(
            f"Evolution: {total} competitive models (+{len(protected)} protected), "
            f"eliminating {elimination_count}, keeping {len(survivors)}"
        )

        # Mark eliminated models in DB
        db = get_session(self.config.db_path)
        eliminated_ids = []
        survivor_ids = []

        try:
            for m in eliminated:
                model = db.query(TradingModel).get(m.model_id)
                if model:
                    model.status = ModelStatus.ELIMINATED
                    model.eliminated_at = datetime.utcnow()
                    eliminated_ids.append(m.model_id)

            for m in survivors:
                survivor_ids.append(m.model_id)

            db.commit()
        finally:
            db.close()

        # Generate offspring to fill the pool
        offspring_needed = self.config.arena.model_count - len(survivors)
        offspring = self._generate_offspring(survivors, offspring_needed)

        # Determine generation number
        db = get_session(self.config.db_path)
        try:
            max_gen = (
                db.query(TradingModel.generation)
                .order_by(TradingModel.generation.desc())
                .first()
            )
            next_gen = (max_gen[0] + 1) if max_gen else 2
        finally:
            db.close()

        # Create offspring models in DB
        offspring_ids = self._persist_offspring(offspring, next_gen)

        # Log generation record
        protected_ids = [m.model_id for m in protected]
        all_model_ids = survivor_ids + offspring_ids + protected_ids
        self._log_generation(
            generation_number=next_gen,
            session_date=session_date,
            model_ids=all_model_ids,
            eliminated_ids=eliminated_ids,
            survivor_ids=survivor_ids,
            offspring_ids=offspring_ids,
            leaderboard=leaderboard,
        )

        # Update COLLAB eligible voters: top 5 performers' strategy types
        top_5_types = list(dict.fromkeys(
            m.strategy_type for m in competitive[:5]
            if m.strategy_type != "collab"
        ))
        if top_5_types:
            db = get_session(self.config.db_path)
            try:
                collab_models = (
                    db.query(TradingModel)
                    .filter(
                        TradingModel.strategy_type == "collab",
                        TradingModel.status == ModelStatus.ACTIVE,
                    )
                    .all()
                )
                for cm in collab_models:
                    params = cm.parameters or {}
                    params["_eligible_voters"] = top_5_types
                    cm.parameters = params
                db.commit()
                logger.info(f"COLLAB voters updated to top 5 types: {top_5_types}")
            finally:
                db.close()

        summary = {
            "generation": next_gen,
            "total_models": total,
            "eliminated": elimination_count,
            "eliminated_names": [m.model_name for m in eliminated],
            "survivors": len(survivors),
            "survivor_names": [m.model_name for m in survivors],
            "offspring": len(offspring_ids),
            "offspring_names": [o["name"] for o in offspring],
            "next_pool_size": len(all_model_ids),
        }

        logger.info(
            f"Generation {next_gen}: "
            f"{len(survivors)} survivors + {len(offspring_ids)} offspring"
        )

        return summary

    def _generate_offspring(
        self,
        survivors: list,
        count: int,
    ) -> list[dict]:
        """Generate offspring via crossover and mutation.

        Strategies:
        - Crossover pairs of survivors weighted by fitness
        - Mutate offspring parameters with bounded perturbation
        - Include at least one pure mutation (no crossover) per generation
        """
        if not survivors or count <= 0:
            return []

        offspring = []

        for i in range(count):
            if len(survivors) >= 2 and i < count - 1:
                # Crossover: pick two parents weighted by fitness
                parent_a, parent_b = self._select_parents(survivors)
                child_params = self._crossover(parent_a, parent_b)
                child_params = self._mutate(child_params)

                # Determine strategy type from fitter parent
                strategy_type = parent_a.strategy_type
                parent_ids = [parent_a.model_id, parent_b.model_id]
                genetic_op = "crossover+mutation"
            else:
                # Pure mutation of a random survivor
                parent = random.choice(survivors)
                child_params = copy.deepcopy(
                    self._get_model_params(parent.model_id)
                )
                child_params = self._mutate(child_params, scale=1.5)  # stronger mutation

                strategy_type = parent.strategy_type
                parent_ids = [parent.model_id]
                genetic_op = "mutation_only"

            offspring.append({
                "name": f"{strategy_type}_gen_next_{i+1}",
                "strategy_type": strategy_type,
                "parameters": child_params,
                "parent_ids": parent_ids,
                "genetic_operation": genetic_op,
            })

        return offspring

    def _select_parents(self, survivors: list) -> tuple:
        """Select two parents via fitness-weighted sampling."""
        if len(survivors) < 2:
            return survivors[0], survivors[0]

        # Weight by fitness (higher is better)
        fitnesses = [
            max(0.01, m.fitness.composite if m.fitness else 0.01)
            for m in survivors
        ]
        total = sum(fitnesses)
        weights = [f / total for f in fitnesses]

        # Weighted selection without replacement
        indices = list(range(len(survivors)))
        parent_a_idx = random.choices(indices, weights=weights, k=1)[0]

        # Remove parent A from pool for second selection
        remaining_indices = [i for i in indices if i != parent_a_idx]
        remaining_weights = [weights[i] for i in remaining_indices]
        total_remaining = sum(remaining_weights)
        if total_remaining > 0:
            remaining_weights = [w / total_remaining for w in remaining_weights]
        else:
            remaining_weights = [1.0 / len(remaining_indices)] * len(remaining_indices)

        parent_b_idx = random.choices(remaining_indices, weights=remaining_weights, k=1)[0]

        return survivors[parent_a_idx], survivors[parent_b_idx]

    def _crossover(self, parent_a, parent_b) -> dict:
        """Blend parameters from two parents.

        For numeric parameters: weighted average based on relative fitness.
        For non-numeric: take from the fitter parent.
        """
        params_a = self._get_model_params(parent_a.model_id)
        params_b = self._get_model_params(parent_b.model_id)

        if not params_a or not params_b:
            return params_a or params_b or {}

        # Compute blend weight from fitness
        fit_a = parent_a.fitness.composite if parent_a.fitness else 0.5
        fit_b = parent_b.fitness.composite if parent_b.fitness else 0.5
        total_fit = fit_a + fit_b
        weight_a = fit_a / total_fit if total_fit > 0 else 0.5

        child = {}
        all_keys = set(list(params_a.keys()) + list(params_b.keys()))

        for key in all_keys:
            val_a = params_a.get(key)
            val_b = params_b.get(key)

            if val_a is None:
                child[key] = val_b
            elif val_b is None:
                child[key] = val_a
            elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # Arithmetic blend
                blended = val_a * weight_a + val_b * (1 - weight_a)
                # Preserve int type if both are int
                if isinstance(val_a, int) and isinstance(val_b, int):
                    child[key] = int(round(blended))
                else:
                    child[key] = blended
            else:
                # Non-numeric: take from fitter parent
                child[key] = val_a if weight_a >= 0.5 else val_b

        return child

    def _mutate(self, params: dict, scale: float = 1.0) -> dict:
        """Apply bounded perturbation to parameters.

        Each numeric parameter has a chance of being mutated by
        +/- mutation_range * scale of its current value.
        """
        mutation_range = self.config.arena.mutation_range * scale
        mutated = {}

        for key, value in params.items():
            if isinstance(value, (int, float)) and random.random() < 0.5:
                # Mutate this parameter
                perturbation = random.uniform(-mutation_range, mutation_range)
                new_value = value * (1 + perturbation)

                if isinstance(value, int):
                    new_value = max(1, int(round(new_value)))
                else:
                    new_value = round(new_value, 6)

                mutated[key] = new_value
            else:
                mutated[key] = value

        return mutated

    def _get_model_params(self, model_id: int) -> dict:
        """Fetch parameters for a model from DB."""
        db = get_session(self.config.db_path)
        try:
            model = db.query(TradingModel).get(model_id)
            return model.parameters if model else {}
        finally:
            db.close()

    def _persist_offspring(self, offspring: list[dict], generation: int) -> list[int]:
        """Write offspring models to the database. Returns list of new model IDs."""
        db = get_session(self.config.db_path)
        ids = []
        try:
            for o in offspring:
                model = TradingModel(
                    name=o["name"].replace("gen_next", f"gen{generation}"),
                    strategy_type=o["strategy_type"],
                    parameters=o["parameters"],
                    generation=generation,
                    parent_ids=o["parent_ids"],
                    genetic_operation=o["genetic_operation"],
                    initial_capital=self.config.arena.initial_capital,
                    current_capital=self.config.arena.initial_capital,
                    mutation_memory=None,
                )
                db.add(model)
                db.flush()
                ids.append(model.id)

            # Validate offspring by instantiating strategies
            for model_id in ids:
                model = db.query(TradingModel).get(model_id)
                strategy = create_strategy(
                    model.strategy_type,
                    model.name,
                    params=model.parameters,
                )
                # set_params normalizes/clamps values
                strategy.set_params(model.parameters)
                model.parameters = strategy.get_params()

            db.commit()
            logger.info(f"Persisted {len(ids)} offspring for generation {generation}")
        except Exception:
            db.rollback()
            logger.exception("Failed to persist offspring")
        finally:
            db.close()

        return ids

    def _log_generation(
        self,
        generation_number: int,
        session_date: str,
        model_ids: list[int],
        eliminated_ids: list[int],
        survivor_ids: list[int],
        offspring_ids: list[int],
        leaderboard: list,
    ) -> None:
        """Record the generation in the database."""
        fitnesses = [
            m.fitness.composite if m.fitness else 0 for m in leaderboard
        ]

        db = get_session(self.config.db_path)
        try:
            record = GenerationRecord(
                generation_number=generation_number,
                session_date=session_date,
                model_ids=model_ids,
                eliminated_ids=eliminated_ids,
                survivor_ids=survivor_ids,
                offspring_ids=offspring_ids,
                best_fitness=max(fitnesses) if fitnesses else None,
                avg_fitness=float(np.mean(fitnesses)) if fitnesses else None,
            )
            db.add(record)
            db.commit()
            logger.info(f"Generation {generation_number} logged")
        except Exception:
            db.rollback()
            logger.exception("Failed to log generation")
        finally:
            db.close()
