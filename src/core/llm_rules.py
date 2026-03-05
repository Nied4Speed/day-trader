"""LLM-generated watch rules via Claude Haiku.

Calls the Anthropic API during self-improvement to generate structured
watch rules for each model. Rules are validated before use.
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_RULE_FORMAT_SPEC = """\
Each rule is a JSON object with these fields:

{
  "watch_when": {
    "indicator": "<indicator_name>",
    "op": "<lt|gt|lte|gte|between>",
    "value": <number or [low, high] for between>
  },
  "entry_when": {
    "price_op": "<lt|gt|lte|gte>",
    "context_key": "<key_name>"
  },
  "context_values": {
    "<key_name>": "<indicator_name>"
  },
  "ttl_bars": <1-20>,
  "reason": "<short description>"
}

- watch_when: Checked each bar against the model's indicators. If true, a watch is created.
- entry_when: Checked at quote frequency (sub-second). Compares the current mid_price against a frozen indicator value.
- context_values: Maps context keys to indicator names. Values are frozen when the watch starts.
- ttl_bars: How many bars before the watch expires without entry (1-20).
- reason: Human-readable explanation.

Available comparison ops: lt (less than), gt (greater than), lte (<=), gte (>=), between ([low, high] inclusive).
For entry_when, price_op compares mid_price against the context value."""


def generate_watch_rules(
    strategy_type: str,
    model_name: str,
    current_params: dict,
    current_rules: list[dict],
    reflection: str,
    available_indicators: list[str],
    performance: dict,
    model_id: str = "claude-haiku-4-5-20251001",
    timeout_sec: int = 10,
    max_rules: int = 3,
) -> Optional[list[dict]]:
    """Generate watch rules for a model using Claude Haiku.

    Args:
        strategy_type: The strategy family (e.g. "breakout").
        model_name: The model's name.
        current_params: Current tunable parameters.
        current_rules: Existing watch rules (may be empty on first run).
        reflection: Post-session reflection text.
        available_indicators: List of indicator names this strategy exposes.
        performance: Dict with return_pct, sharpe, trades, win_rate, fitness.
        model_id: Anthropic model to use.
        timeout_sec: API timeout.
        max_rules: Maximum number of rules to generate.

    Returns:
        List of validated rule dicts, or None on failure (keep existing rules).
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, skipping watch rule generation")
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, skipping watch rule generation")
        return None

    from src.core.watch_rules import validate_rule

    prompt = _build_prompt(
        strategy_type, model_name, current_params, current_rules,
        reflection, available_indicators, performance, max_rules,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout_sec)
        response = client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Extract JSON array from response
        rules = _parse_rules_json(text)
        if rules is None:
            logger.warning(f"Failed to parse LLM rules for {model_name}")
            return None

        # Validate each rule
        validated = []
        all_indicators = available_indicators + ["has_position"]
        for i, rule in enumerate(rules):
            if len(validated) >= max_rules:
                break
            valid, err = validate_rule(rule, all_indicators)
            if valid:
                validated.append(rule)
            else:
                logger.debug(f"Rule {i} for {model_name} invalid: {err}")

        logger.info(
            f"  {model_name}: LLM generated {len(rules)} rules, "
            f"{len(validated)} valid"
        )
        return validated

    except Exception as e:
        logger.warning(f"LLM watch rule generation failed for {model_name}: {e}")
        return None


def _build_prompt(
    strategy_type: str,
    model_name: str,
    current_params: dict,
    current_rules: list[dict],
    reflection: str,
    available_indicators: list[str],
    performance: dict,
    max_rules: int,
) -> str:
    # Filter out non-scalar params for readability
    display_params = {
        k: v for k, v in current_params.items()
        if isinstance(v, (int, float, str, bool))
    }

    parts = [
        f"You are generating watch rules for a trading model in an evolutionary arena.",
        f"",
        f"## Model",
        f"- Name: {model_name}",
        f"- Strategy type: {strategy_type}",
        f"- Parameters: {json.dumps(display_params, indent=2)}",
        f"",
        f"## Available Indicators",
        f"These values are computed each bar and available for watch rules:",
        f"{', '.join(available_indicators + ['has_position'])}",
        f"",
        f"## Performance This Session",
        f"- Return: {performance.get('return_pct', 0):+.3f}%",
        f"- Sharpe: {performance.get('sharpe', 0):.2f}",
        f"- Trades: {performance.get('trades', 0)}",
        f"- Win rate: {performance.get('win_rate', 0):.1f}%",
        f"- Fitness: {performance.get('fitness', 0):.4f}",
        f"",
    ]

    if reflection:
        parts.extend([
            f"## Session Reflection",
            reflection,
            f"",
        ])

    if current_rules:
        parts.extend([
            f"## Current Watch Rules",
            json.dumps(current_rules, indent=2),
            f"",
            f"You may keep, modify, or replace these rules based on performance.",
            f"",
        ])
    else:
        parts.extend([
            f"## Current Watch Rules",
            f"None — this is the first time generating rules for this model.",
            f"Generate sensible defaults for a {strategy_type} strategy.",
            f"",
        ])

    parts.extend([
        f"## Rule Format",
        _RULE_FORMAT_SPEC,
        f"",
        f"## Instructions",
        f"Generate up to {max_rules} watch rules for this model. Watch rules let the model",
        f"monitor symbols at quote frequency when a setup is forming but not yet ready.",
        f"",
        f"Think about what conditions indicate a potential entry is near for a {strategy_type}",
        f"strategy. The watch_when condition should identify \"almost ready\" setups,",
        f"and entry_when should specify the exact price level that triggers a buy.",
        f"",
        f"Rules should use the available indicators listed above. Be specific with",
        f"numeric values — use values that make sense for the strategy type and current params.",
        f"",
        f"Respond with ONLY a JSON array of rule objects. No explanation, no markdown fences.",
    ])

    return "\n".join(parts)


def _parse_rules_json(text: str) -> Optional[list[dict]]:
    """Extract a JSON array of rules from LLM response text."""
    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    for fence in ["```json", "```"]:
        if fence in text:
            start = text.index(fence) + len(fence)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                result = json.loads(text[start:end].strip())
                if isinstance(result, list):
                    return result
                if isinstance(result, dict):
                    return [result]
            except (json.JSONDecodeError, ValueError):
                pass

    # Try finding array brackets
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None
