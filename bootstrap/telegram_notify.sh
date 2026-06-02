#!/usr/bin/env bash
# Push a message to Telegram. Usage: ./telegram_notify.sh "your message"
# Reads TG_TOKEN and TG_CHAT from ~/.telegram_creds (keep that file private).
set -uo pipefail
[ -f "$HOME/.telegram_creds" ] && source "$HOME/.telegram_creds"
: "${TG_TOKEN:?set TG_TOKEN in ~/.telegram_creds}"
: "${TG_CHAT:?set TG_CHAT in ~/.telegram_creds}"
MSG="${1:-(no message)}"
curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
  -d chat_id="${TG_CHAT}" \
  -d parse_mode="Markdown" \
  --data-urlencode text="${MSG}" >/dev/null && echo "sent" || echo "failed"
