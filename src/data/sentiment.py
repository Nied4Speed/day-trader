"""Keyword-based news sentiment scorer.

Fast, deterministic sentiment scoring using curated word lists.
Returns a float in [-1.0, +1.0] where positive = bullish, negative = bearish.
Designed to be swappable with a model-based scorer later.
"""

import re

# Bullish keywords (financial context)
BULLISH_WORDS = {
    # Strong bullish
    "surge", "surges", "surging", "soar", "soars", "soaring",
    "rally", "rallies", "rallying", "boom", "booming",
    "skyrocket", "skyrockets", "breakout", "moonshot",
    # Moderate bullish
    "gain", "gains", "gaining", "rise", "rises", "rising",
    "jump", "jumps", "jumping", "climb", "climbs", "climbing",
    "advance", "advances", "advancing", "upgrade", "upgrades",
    "beat", "beats", "beating", "exceed", "exceeds", "exceeding",
    "outperform", "outperforms", "bullish", "optimistic",
    "record", "high", "growth", "strong", "strength",
    "positive", "upbeat", "robust", "boost", "momentum",
    "buy", "overweight", "upside", "recovery", "rebound",
    "profit", "profitable", "revenue", "earnings",
}

# Bearish keywords (financial context)
BEARISH_WORDS = {
    # Strong bearish
    "crash", "crashes", "crashing", "plunge", "plunges", "plunging",
    "collapse", "collapses", "collapsing", "tank", "tanks", "tanking",
    "plummet", "plummets", "plummeting", "freefall",
    # Moderate bearish
    "fall", "falls", "falling", "drop", "drops", "dropping",
    "decline", "declines", "declining", "slip", "slips", "slipping",
    "slide", "slides", "sliding", "sink", "sinks", "sinking",
    "downgrade", "downgrades", "miss", "misses", "missing",
    "underperform", "underperforms", "bearish", "pessimistic",
    "low", "weak", "weakness", "negative", "concern", "concerns",
    "risk", "risks", "warning", "sell", "underweight", "downside",
    "loss", "losses", "deficit", "recession", "layoff", "layoffs",
    "cut", "cuts", "cutting", "debt", "default", "bankruptcy",
    "investigation", "lawsuit", "fraud", "scandal", "probe",
}

# Intensity modifiers
AMPLIFIERS = {"very", "extremely", "sharply", "dramatically", "significantly", "massive", "huge"}
DAMPENERS = {"slightly", "somewhat", "modestly", "marginally", "gradually"}

_WORD_RE = re.compile(r"[a-z]+")


def score_text(text: str) -> float:
    """Score a text string for financial sentiment.

    Returns a float in [-1.0, +1.0].
    """
    if not text:
        return 0.0

    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0

    score = 0.0
    prev_word = ""

    for word in words:
        if word in BULLISH_WORDS:
            weight = 1.0
            if prev_word in AMPLIFIERS:
                weight = 1.5
            elif prev_word in DAMPENERS:
                weight = 0.5
            score += weight
        elif word in BEARISH_WORDS:
            weight = 1.0
            if prev_word in AMPLIFIERS:
                weight = 1.5
            elif prev_word in DAMPENERS:
                weight = 0.5
            score -= weight
        prev_word = word

    # Normalize: cap at [-1, 1] using tanh-like scaling
    if score == 0:
        return 0.0
    # Scale so ~3 keywords ≈ 0.7 score
    normalized = score / (abs(score) + 2.0)
    return max(-1.0, min(1.0, normalized))


def score_article(headline: str, summary: str = "") -> float:
    """Score a news article. Headline weighted 70%, summary 30%."""
    h_score = score_text(headline)
    if not summary:
        return h_score
    s_score = score_text(summary)
    return h_score * 0.7 + s_score * 0.3
