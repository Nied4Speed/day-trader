"""Watch rule DSL: evaluate structured JSON rules against indicator values.

Rules are simple dicts that avoid eval() entirely. The LLM generates them,
and this module evaluates them safely at bar-time and quote-time.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Maps strategy type -> list of indicators exposed by get_indicators()
INDICATOR_CATALOG: dict[str, list[str]] = {
    "breakout": ["close", "range_high", "range_low", "threshold_high", "threshold_low", "distance_to_breakout_pct"],
    "momentum": ["close", "roc", "ref_price", "volume_ratio"],
    "rsi_reversion": ["close", "rsi"],
    "bollinger_bands": ["close", "lower_band", "upper_band", "band_width", "band_position"],
    "ma_crossover": ["close", "fast_ma", "slow_ma", "ma_spread"],
    "macd": ["close", "histogram", "prev_histogram", "macd_line", "signal_line"],
    "vwap_reversion": ["close", "vwap", "deviation_from_vwap"],
    "stochastic": ["close", "k", "d", "prev_k", "prev_d"],
    "mean_reversion": ["close", "ma", "deviation_from_ma"],
    "ml_predictor": ["close", "predicted_return"],
    "news_sentiment": ["close", "sentiment"],
}

# All strategies also get "has_position" from base class
_BASE_INDICATORS = ["has_position"]

VALID_OPS = {"lt", "gt", "lte", "gte", "between"}
VALID_PRICE_OPS = {"lt", "gt", "lte", "gte"}


def _compare(value: float, op: str, target: Any) -> bool:
    """Apply a comparison operator."""
    if op == "lt":
        return value < target
    elif op == "gt":
        return value > target
    elif op == "lte":
        return value <= target
    elif op == "gte":
        return value >= target
    elif op == "between":
        if not isinstance(target, (list, tuple)) or len(target) != 2:
            return False
        return target[0] <= value <= target[1]
    return False


def evaluate_watch_condition(rule: dict, indicators: dict[str, float]) -> bool:
    """Check if a watch rule's watch_when condition is met.

    Args:
        rule: A watch rule dict with "watch_when" key.
        indicators: Current indicator values from get_indicators().

    Returns:
        True if the condition is satisfied and a watch should be created.
    """
    condition = rule.get("watch_when")
    if not condition:
        return False

    indicator_name = condition.get("indicator")
    op = condition.get("op")
    target = condition.get("value")

    if not indicator_name or not op or target is None:
        return False

    value = indicators.get(indicator_name)
    if value is None:
        return False

    return _compare(value, op, target)


def build_watch_context(rule: dict, indicators: dict[str, float]) -> dict[str, float]:
    """Freeze indicator values into the watch context for later quote-time evaluation.

    Args:
        rule: A watch rule dict with "context_values" mapping.
        indicators: Current indicator values.

    Returns:
        Dict of context_key -> frozen indicator value.
    """
    context: dict[str, float] = {}
    context_spec = rule.get("context_values", {})

    for context_key, indicator_name in context_spec.items():
        value = indicators.get(indicator_name)
        if value is not None:
            context[context_key] = value

    return context


def evaluate_entry_condition(rule: dict, mid_price: float, context: dict[str, float]) -> bool:
    """Check if a watch rule's entry condition is met at quote time.

    Args:
        rule: A watch rule dict with "entry_when" key.
        mid_price: Current mid price from quote.
        context: Frozen context values from build_watch_context().

    Returns:
        True if entry should happen.
    """
    condition = rule.get("entry_when")
    if not condition:
        return False

    price_op = condition.get("price_op")
    context_key = condition.get("context_key")

    if not price_op or not context_key:
        return False

    target_value = context.get(context_key)
    if target_value is None:
        return False

    return _compare(mid_price, price_op, target_value)


def validate_rule(rule: dict, available_indicators: list[str]) -> tuple[bool, str]:
    """Validate a watch rule against the available indicators for a strategy type.

    Returns:
        (is_valid, error_message). error_message is "" if valid.
    """
    if not isinstance(rule, dict):
        return False, "rule is not a dict"

    # Check watch_when
    watch_when = rule.get("watch_when")
    if not isinstance(watch_when, dict):
        return False, "missing or invalid watch_when"

    indicator = watch_when.get("indicator")
    if not indicator or indicator not in available_indicators:
        return False, f"unknown indicator '{indicator}', available: {available_indicators}"

    op = watch_when.get("op")
    if op not in VALID_OPS:
        return False, f"invalid op '{op}', must be one of {VALID_OPS}"

    value = watch_when.get("value")
    if value is None:
        return False, "watch_when.value is required"
    if op == "between":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False, "between op requires value=[low, high]"
        if not all(isinstance(v, (int, float)) for v in value):
            return False, "between values must be numeric"
    else:
        if not isinstance(value, (int, float)):
            return False, f"value must be numeric for op '{op}'"

    # Check entry_when
    entry_when = rule.get("entry_when")
    if not isinstance(entry_when, dict):
        return False, "missing or invalid entry_when"

    price_op = entry_when.get("price_op")
    if price_op not in VALID_PRICE_OPS:
        return False, f"invalid price_op '{price_op}', must be one of {VALID_PRICE_OPS}"

    context_key = entry_when.get("context_key")
    if not context_key or not isinstance(context_key, str):
        return False, "entry_when.context_key is required"

    # Check context_values references valid indicators
    context_values = rule.get("context_values", {})
    if not isinstance(context_values, dict):
        return False, "context_values must be a dict"

    for ckey, ind_name in context_values.items():
        if ind_name not in available_indicators:
            return False, f"context_values['{ckey}'] references unknown indicator '{ind_name}'"

    # context_key used in entry_when must be defined in context_values
    if context_key not in context_values:
        return False, f"entry_when.context_key '{context_key}' not found in context_values"

    # Check ttl_bars
    ttl = rule.get("ttl_bars", 5)
    if not isinstance(ttl, (int, float)) or ttl < 1 or ttl > 20:
        return False, f"ttl_bars must be 1-20, got {ttl}"

    # Check reason
    reason = rule.get("reason", "")
    if not isinstance(reason, str):
        return False, "reason must be a string"

    return True, ""
