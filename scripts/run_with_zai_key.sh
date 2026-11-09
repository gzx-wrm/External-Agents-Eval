#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

read -r -s -p "Z.AI API key: " ZAI_API_KEY
echo >&2
if [[ -z "${ZAI_API_KEY}" ]]; then
  echo "Z.AI API key cannot be empty" >&2
  exit 2
fi

export ZAI_API_KEY
exec "$@"
