"""One-shot script to backfill alpaca_pnl on existing session records."""

import sys
sys.path.insert(0, ".")

from src.core.database import init_db, get_session, SessionRecord

init_db()
db = get_session()

# Known Alpaca P&L values
backfill = {
    "2026-03-10": -3.62,
    # Add more dates here if known, e.g.:
    # "2026-03-09": <value>,
}

for date, pnl in backfill.items():
    sessions = (
        db.query(SessionRecord)
        .filter(SessionRecord.session_date == date)
        .all()
    )
    if not sessions:
        print(f"  {date}: no session records found")
        continue
    # Put the full day P&L on the last session of the day
    last = max(sessions, key=lambda s: s.session_number)
    last.alpaca_pnl = pnl
    print(f"  {date}: set session #{last.session_number} alpaca_pnl = {pnl}")

db.commit()
db.close()
print("Done.")
