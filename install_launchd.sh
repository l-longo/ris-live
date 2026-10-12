#!/bin/bash
# Installs an hourly launchd job that runs run_update.sh with the current python3.
REPO="$(cd "$(dirname "$0")" && pwd)"
PY="${RIS_PYTHON:-/opt/anaconda3/bin/python}"
PLIST="$HOME/Library/LaunchAgents/eu.jrc.reddit-inflation-signal.plist"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>eu.jrc.reddit-inflation-signal</string>
  <key>ProgramArguments</key><array><string>$REPO/run_update.sh</string></array>
  <key>EnvironmentVariables</key><dict><key>RIS_PYTHON</key><string>$PY</string><key>PATH</key><string>$(dirname "$PY"):/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string></dict>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$REPO/logs/launchd.out</string>
  <key>StandardErrorPath</key><string>$REPO/logs/launchd.err</string>
</dict></plist>
PL
mkdir -p "$REPO/logs"
launchctl unload "$PLIST" 2>/dev/null
launchctl load "$PLIST" && echo "Installed: runs every hour with $PY. Log: $REPO/logs/update.log"
