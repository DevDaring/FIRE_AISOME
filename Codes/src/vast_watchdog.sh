#!/usr/bin/env bash
# Destroy a Vast.ai instance as soon as the API becomes reachable again.
#
# Why this exists: on 2026-08-03 vast.ai's own DNS went NXDOMAIN at the
# authoritative level (Google's public resolver returned Status 3 for the apex
# domain), so `vast_gpu.py down` could not reach the API — while the rented GPU
# kept billing by the second. This loop retries until it lands, so an outage on
# their side cannot quietly drain the account.
#
# Usage:  bash src/vast_watchdog.sh <instance_id> [max_minutes]
# Safe to run when nothing is rented: destroying a dead id is a no-op.
set -uo pipefail
cd "$(dirname "$0")/.."

IID="${1:?usage: bash src/vast_watchdog.sh <instance_id> [max_minutes]}"
MAX_MIN="${2:-720}"                      # give up after 12h and shout
KEY=$(grep -oP '^VAST_AI_API_KEY=\K.*' .env | tr -d '"'"'"'" ' | head -1)
LOG=logs/vast_watchdog.log
mkdir -p logs

say() { echo "$(date -u '+%F %T UTC')  $*" | tee -a "$LOG"; }

say "watchdog started for instance $IID (will retry for up to ${MAX_MIN}m)"
DEADLINE=$(( $(date +%s) + MAX_MIN * 60 ))

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # DELETE is idempotent: 200 on a live instance, error on an already-dead one.
    CODE=$(curl -s --max-time 25 -o /tmp/vw.json -w '%{http_code}' \
        -X DELETE -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
        -d '{}' "https://console.vast.ai/api/v1/instances/$IID/" 2>/dev/null) || true
    CODE="${CODE: -3}"        # curl already writes 000 on failure; keep one copy

    if [ "$CODE" = "200" ]; then
        say "DESTROYED instance $IID — billing stopped"
        rm -f artifacts/vast_instance.json
        exit 0
    fi
    if [ "$CODE" = "404" ]; then
        say "instance $IID no longer exists — nothing billing"
        rm -f artifacts/vast_instance.json
        exit 0
    fi
    if [ "$CODE" = "000" ]; then
        say "vast.ai unreachable (DNS/network) — retrying in 120s"
    else
        say "unexpected http $CODE: $(head -c 200 /tmp/vw.json)"
    fi
    sleep 120
done

say "!!! GAVE UP after ${MAX_MIN}m — instance $IID MAY STILL BE BILLING."
say "!!! Destroy it by hand at https://cloud.vast.ai/instances/"
exit 1
