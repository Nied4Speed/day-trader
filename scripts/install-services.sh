#!/bin/bash
# Install launchd services for Day Trader Arena
# - Backend auto-starts on login and restarts on crash
# - Arena auto-starts at 9:00 AM ET on weekdays
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

mkdir -p "$LAUNCH_AGENTS"
mkdir -p "$SCRIPT_DIR/../logs"

# Show timezone info
LOCAL_TZ=$(date +%Z)
echo "Mac timezone: $LOCAL_TZ"
echo "Arena scheduled for 6:00 AM PT / 9:00 AM ET weekdays"
if [[ "$LOCAL_TZ" != "PDT" && "$LOCAL_TZ" != "PST" ]]; then
    echo "NOTE: Plist is configured for Pacific Time (Hour=6)."
    echo "If your Mac isn't on PT, adjust the Hour in com.daytrader.arena-start.plist."
    echo ""
fi

# Stop existing services if loaded
launchctl bootout gui/$(id -u)/com.daytrader.backend 2>/dev/null || true
launchctl bootout gui/$(id -u)/com.daytrader.arena-start 2>/dev/null || true

# Copy plists
cp "$SCRIPT_DIR/com.daytrader.backend.plist" "$LAUNCH_AGENTS/"
cp "$SCRIPT_DIR/com.daytrader.arena-start.plist" "$LAUNCH_AGENTS/"

# Load services
launchctl bootstrap gui/$(id -u) "$LAUNCH_AGENTS/com.daytrader.backend.plist"
launchctl bootstrap gui/$(id -u) "$LAUNCH_AGENTS/com.daytrader.arena-start.plist"

echo ""
echo "Services installed:"
echo "  1. com.daytrader.backend    — uvicorn on :8000 (starts now, restarts on crash)"
echo "  2. com.daytrader.arena-start — arena launch at 6:00 AM PT / 9:00 AM ET weekdays"
echo ""
echo "Dashboard: http://localhost:8000"
echo ""
echo "Management commands:"
echo "  launchctl kickstart gui/$(id -u)/com.daytrader.backend    # force restart backend"
echo "  launchctl kill SIGTERM gui/$(id -u)/com.daytrader.backend # stop backend"
echo "  tail -f logs/arena.log                                     # watch arena logs"
echo ""
echo "To uninstall: bash scripts/uninstall-services.sh"
