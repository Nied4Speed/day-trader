#!/bin/bash
# Run the trading arena session, then evolve after it ends
cd /Users/brandonnied/day-trader
source .venv/bin/activate

echo "$(date): Starting arena session" >> logs/arena.log
python main.py run >> logs/arena.log 2>&1

echo "$(date): Running evolution" >> logs/arena.log
python main.py evolve >> logs/arena.log 2>&1

echo "$(date): Session complete" >> logs/arena.log
