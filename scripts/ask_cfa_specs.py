"""Ask CFA for exact strategy specs with full review context."""
import json
import os

from dotenv import load_dotenv

load_dotenv(".env")

import anthropic

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=180)

# Load the full CFA review
with open("logs/cfa_review_2026-03-11.md") as f:
    review_text = f.read()

prompt = f"""You are the Chief Strategy Architect for an evolutionary trading arena. You just
completed your daily review for 2026-03-11. Here is your FULL review with all the data and
analysis you produced:

---
{review_text}
---

Based on YOUR OWN analysis above, you recommended 3 specific actions:

1. **Replace vol_profile_gen1_20** — it lost -$49.98 with 59% win rate. Volume profile complexity
   failed. You said to replace with a momentum breakout.
2. **Replace vol_compress_gen1_14** — it lost -$32.44 with 15% win rate. Volatility compression
   breakout signals were false positives. You said to replace with simpler ATR breakout.
3. **Rewrite cfa_analyst** — 0% win rate, -$32.38. Your own model is failing. You said to rebuild
   it with the proven VWAP mean reversion logic that powers the 90%+ win rate models.

The proven patterns FROM YOUR DATA:
- vwap_reversion models: 88-94% win rate with deviation threshold 0.001
- rsi_reversion_gen9: 78-79% win rate with evolved parameters
- bollinger: 91% win rate
- Average win $0.17 vs average loss $0.87 — expectancy is negative despite high win rates
- High frequency trading (500+ trades) with tight stops outperforms low frequency
- HIMS +$83.45, PATH +$20.80, IWM +$14.73 were best symbols
- DRIP -$61.83, ORCX -$31.46, DOMO -$20.66 were worst

Strategy base class API:
- on_bar(self, bar) -> TradeSignal or None
- bar has: symbol, open, high, low, close, volume, timestamp, vwap
- self._bar_history[symbol] = list of recent BarData
- self.get_close_series(symbol) -> pandas Series of closes
- self._positions[symbol] -> current qty, self._entry_prices[symbol] -> entry price
- self.compute_quantity(price, alloc, symbol=symbol) -> position-aware qty
- self.record_bar(bar) + self.check_liquidation(bar) must be called first
- adapt(self, recent_signals, recent_fills, realized_pnl) every 15 bars
- Risk params on base: stop_loss_pct, take_profit_pct, trailing_stop_tiers, patience_stop_tiers

For EACH of the 3 strategies, provide the EXACT specification. Every single parameter with its
exact numeric value. Entry/exit logic with specific thresholds. Adapt logic with bounds.
These specs get implemented directly as code — nothing vague.

Return a JSON array of exactly 3 objects with these fields:
- strategy_name (string)
- strategy_type (snake_case registry key)
- description (one line)
- parameters (object with ALL params and exact numeric values)
- entry_logic (exact conditions with specific numbers)
- exit_logic (exact conditions with specific numbers)
- adapt_logic (what to tune, direction, bounds)
- stop_loss_pct (number)
- take_profit_pct (number)
- trailing_stop_tiers (array of [threshold, distance] pairs)
- patience_stop_tiers (array of [bars, exit_pct] pairs)
- allocation_pct (number)
- max_positions (number)
- predicted_win_rate (number 0-1)
- predicted_avg_pnl (number in dollars)

Return ONLY the JSON array. No markdown, no explanation."""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8192,
    messages=[{"role": "user", "content": prompt}],
)

text = response.content[0].text.strip()
print(text)

# Also save to file for reference
with open("logs/cfa_strategy_specs_2026-03-11.json", "w") as f:
    # Try to parse and pretty-print
    try:
        parsed = json.loads(text)
        json.dump(parsed, f, indent=2)
        print("\n\nParsed and saved to logs/cfa_strategy_specs_2026-03-11.json")
    except json.JSONDecodeError:
        f.write(text)
        print("\n\nSaved raw text to logs/cfa_strategy_specs_2026-03-11.json")
