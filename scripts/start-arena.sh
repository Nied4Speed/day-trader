#!/bin/bash
# Wait for backend to be ready, then start the arena.
# Called by launchd at 6:00 AM PT / 9:00 AM ET on weekdays.

URL="http://localhost:8000/api/arena/start"
MAX_WAIT=60  # seconds to wait for backend

for i in $(seq 1 $MAX_WAIT); do
    if curl -s -o /dev/null -w '' http://localhost:8000/api/arena/state 2>/dev/null; then
        echo "$(date): Backend ready, starting arena..."
        curl -s -X POST -H "Content-Type: application/json" \
            -d '{"num_sessions": 1, "session_minutes": 390}' \
            "$URL"
        echo ""
        exit 0
    fi
    sleep 1
done

echo "$(date): Backend not ready after ${MAX_WAIT}s, giving up"
exit 1
