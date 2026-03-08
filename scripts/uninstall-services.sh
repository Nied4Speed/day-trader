#!/bin/bash
# Remove Day Trader Arena launchd services
set -e

LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

launchctl bootout gui/$(id -u)/com.daytrader.backend 2>/dev/null || true
launchctl bootout gui/$(id -u)/com.daytrader.arena-start 2>/dev/null || true

rm -f "$LAUNCH_AGENTS/com.daytrader.backend.plist"
rm -f "$LAUNCH_AGENTS/com.daytrader.arena-start.plist"

echo "Services removed. Backend and arena auto-start disabled."
