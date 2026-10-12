#!/bin/bash
# Called by launchd every hour: label new posts, rebuild the JSON, push to GitHub Pages.
REPO="$(cd "$(dirname "$0")" && pwd)"
PY="${RIS_PYTHON:-/opt/anaconda3/bin/python}"
LOCK="$REPO/.update.lock"
mkdir -p "$REPO/logs"
if ! mkdir "$LOCK" 2>/dev/null; then echo "$(date) previous run still active, skipping" >> "$REPO/logs/update.log"; exit 0; fi
trap 'rmdir "$LOCK"' EXIT
cd "$REPO"
git pull --rebase --quiet >> "$REPO/logs/update.log" 2>&1
[ -f "$REPO/.env" ] && set -a && . "$REPO/.env" && set +a
PUSH=1 /usr/bin/caffeinate -i "$PY" update_dashboard.py >> "$REPO/logs/update.log" 2>&1
