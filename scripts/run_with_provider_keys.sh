#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <command> [args ...]" >&2
  exit 2
fi

read -r -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
echo >&2
if [[ -z "${DEEPSEEK_API_KEY}" ]]; then
  echo "DeepSeek API key cannot be empty" >&2
  exit 2
fi

read -r -s -p "OpenAI-compatible API key: " OPENAI_API_KEY
echo >&2
if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "OpenAI-compatible API key cannot be empty" >&2
  exit 2
fi

export DEEPSEEK_API_KEY OPENAI_API_KEY
exec "$@"
