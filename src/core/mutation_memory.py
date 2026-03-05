"""Mutation memory: tracks which parameter change directions correlate with improvement.

Each model accumulates a per-parameter tally of up/down successes and failures.
During self-improvement, this tally biases future mutations toward directions
that historically worked, while still maintaining randomness in magnitude.
"""

import random
from typing import Optional


class MutationMemory:
    """Static utilities for mutation memory blobs stored as JSON on TradingModel."""

    DECAY_FACTOR = 0.95  # exponential decay per evaluation cycle
    MIN_OBSERVATIONS = 2  # need this many data points before biasing

    @staticmethod
    def compute_bias(tallies: dict) -> float:
        """Compute directional bias for a single parameter.

        Returns a value in [-1.0, +1.0]:
            Positive = bias toward increasing the param.
            Negative = bias toward decreasing.
            0.0 = no bias (random).
        """
        up_success = tallies.get("up_successes", 0)
        up_fail = tallies.get("up_failures", 0)
        down_success = tallies.get("down_successes", 0)
        down_fail = tallies.get("down_failures", 0)

        total = up_success + up_fail + down_success + down_fail
        if total < MutationMemory.MIN_OBSERVATIONS:
            return 0.0

        up_score = up_success - up_fail
        down_score = down_success - down_fail
        net = up_score - down_score  # positive = "up" worked better

        # Soft normalization: maxes out around +/-0.8
        return net / (abs(net) + 2)

    @staticmethod
    def get_biases(memory: Optional[dict]) -> dict[str, float]:
        """Extract bias for every param from a mutation_memory blob."""
        if not memory:
            return {}
        biases = {}
        for key, tallies in memory.items():
            if key.startswith("_"):
                continue  # skip _pending and other meta keys
            if isinstance(tallies, dict) and "up_successes" in tallies:
                biases[key] = MutationMemory.compute_bias(tallies)
        return biases

    @staticmethod
    def get_observation_count(memory: Optional[dict], param: str) -> int:
        """Get total observations for a param."""
        if not memory or param not in memory:
            return 0
        t = memory[param]
        if not isinstance(t, dict):
            return 0
        return (
            t.get("up_successes", 0) + t.get("up_failures", 0) +
            t.get("down_successes", 0) + t.get("down_failures", 0)
        )

    @staticmethod
    def record_pending(
        memory: Optional[dict],
        pre_return_pct: float,
        mutations: dict[str, str],
    ) -> dict:
        """Record pending mutations to be evaluated after next session.

        Args:
            memory: Current mutation_memory blob (or None).
            pre_return_pct: Return % from the session before mutation.
            mutations: {param_name: "up" | "down"} for each mutated param.

        Returns:
            Updated memory blob with _pending key set.
        """
        if memory is None:
            memory = {}
        memory["_pending"] = {
            "pre_return_pct": pre_return_pct,
            "mutations": mutations,
        }
        return memory

    @staticmethod
    def evaluate_pending(
        memory: Optional[dict],
        post_return_pct: float,
        decay: float = 0.95,
    ) -> dict:
        """Evaluate pending mutations and update tallies.

        Compares post_return_pct to the stored pre_return_pct. For each
        mutated param, records whether the direction correlated with
        improvement. Applies exponential decay to existing tallies.

        Returns:
            Updated memory blob with _pending cleared.
        """
        if not memory or "_pending" not in memory:
            return memory or {}

        pending = memory.pop("_pending")
        pre_return = pending.get("pre_return_pct", 0.0)
        mutations = pending.get("mutations", {})
        improved = post_return_pct > pre_return

        # Apply decay to all existing tallies
        for key, tallies in memory.items():
            if key.startswith("_") or not isinstance(tallies, dict):
                continue
            for field in ("up_successes", "up_failures", "down_successes", "down_failures"):
                if field in tallies:
                    tallies[field] = round(tallies[field] * decay, 3)

        # Update tallies for each mutated param
        for param, direction in mutations.items():
            if param not in memory:
                memory[param] = {
                    "up_successes": 0, "up_failures": 0,
                    "down_successes": 0, "down_failures": 0,
                }
            tallies = memory[param]

            if direction == "up":
                if improved:
                    tallies["up_successes"] = tallies.get("up_successes", 0) + 1
                else:
                    tallies["up_failures"] = tallies.get("up_failures", 0) + 1
            elif direction == "down":
                if improved:
                    tallies["down_successes"] = tallies.get("down_successes", 0) + 1
                else:
                    tallies["down_failures"] = tallies.get("down_failures", 0) + 1

        return memory

    @staticmethod
    def apply_bias(mutation_strength: float, bias: float, dampening: float = 0.6) -> float:
        """Generate a biased perturbation value.

        Args:
            mutation_strength: Base perturbation range (e.g., 0.10 for 10%).
            bias: Directional bias in [-1, +1] from compute_bias().
            dampening: How much to trust the bias (0=ignore, 1=full).

        Returns:
            A perturbation value, biased toward the historically successful direction.
        """
        # Shift center of distribution by bias
        center = bias * mutation_strength * dampening
        # Narrow spread slightly when bias is strong
        spread = mutation_strength * (1 - abs(bias) * 0.3)
        perturbation = center + random.uniform(-spread, spread)
        # Clamp to avoid extreme mutations
        limit = mutation_strength * 1.5
        return max(-limit, min(limit, perturbation))
