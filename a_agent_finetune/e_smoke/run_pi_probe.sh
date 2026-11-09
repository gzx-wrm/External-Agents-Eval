#!/bin/bash
set -euo pipefail

E_SMOKE_SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
E_SMOKE_BASE_DIR=${1:?Provide the frozen repository directory}
E_SMOKE_OUTPUT_DIR=${2:?Provide a new output directory}
E_SMOKE_PROMPT_FILE=${3:-$E_SMOKE_SCRIPT_DIR/tool_probe.txt}
E_SMOKE_SYSTEM_FILE=${4:-}
E_SMOKE_BASE_DIR=$(cd "$E_SMOKE_BASE_DIR" && pwd)
mkdir -p "$E_SMOKE_OUTPUT_DIR"
E_SMOKE_OUTPUT_DIR=$(cd "$E_SMOKE_OUTPUT_DIR" && pwd)
if [ -e "$E_SMOKE_OUTPUT_DIR/pi.jsonl" ]; then
  echo "Output already exists; use a new directory." >&2
  exit 1
fi
mkdir -p "$E_SMOKE_OUTPUT_DIR/sessions" "$E_SMOKE_OUTPUT_DIR/pi-config"
cp "$E_SMOKE_PROMPT_FILE" "$E_SMOKE_OUTPUT_DIR/prompt.txt"
cp "$E_SMOKE_SCRIPT_DIR/models.json" "$E_SMOKE_OUTPUT_DIR/models.json"
cp "$E_SMOKE_SCRIPT_DIR/settings.json" "$E_SMOKE_OUTPUT_DIR/settings.json"
cp "$E_SMOKE_OUTPUT_DIR/models.json" "$E_SMOKE_OUTPUT_DIR/pi-config/models.json"
cp "$E_SMOKE_OUTPUT_DIR/settings.json" "$E_SMOKE_OUTPUT_DIR/pi-config/settings.json"
printf '{}\n' > "$E_SMOKE_OUTPUT_DIR/pi-config/auth.json"
cp "$E_SMOKE_SCRIPT_DIR/inference_options.ts" "$E_SMOKE_OUTPUT_DIR/pi-config/inference_options.ts"
E_SMOKE_SYSTEM_ARGS=()
if [ -n "$E_SMOKE_SYSTEM_FILE" ]; then
  cp "$E_SMOKE_SYSTEM_FILE" "$E_SMOKE_OUTPUT_DIR/system-prompt.txt"
  E_SMOKE_SYSTEM_ARGS=(--system-prompt "$(cat "$E_SMOKE_OUTPUT_DIR/system-prompt.txt")")
fi

E_SMOKE_IMAGE=${E_SMOKE_IMAGE:-external-agents-eval/pi-e-smoke:v0.84.2-arm64}
E_SMOKE_TIMEOUT_SEC=${E_SMOKE_TIMEOUT_SEC:-600}
E_SMOKE_CONTAINER="e-qwen-pi-probe-$(date +%Y%m%d-%H%M%S)-$$"
printf '%s\n' "$E_SMOKE_CONTAINER" > "$E_SMOKE_OUTPUT_DIR/container-name.txt"

if docker run --rm -i --name "$E_SMOKE_CONTAINER" \
  --entrypoint /usr/bin/timeout \
  --workdir /work/base \
  --mount "type=bind,source=$E_SMOKE_BASE_DIR,target=/work/base,readonly" \
  --mount "type=bind,source=$E_SMOKE_OUTPUT_DIR/pi-config,target=/work/config" \
  --mount "type=bind,source=$E_SMOKE_OUTPUT_DIR/sessions,target=/work/sessions" \
  --env PI_CODING_AGENT_DIR=/work/config \
  --env OPENAI_API_KEY=EMPTY \
  --env PI_OFFLINE=1 \
  --env PI_TELEMETRY=0 \
  "$E_SMOKE_IMAGE" "$E_SMOKE_TIMEOUT_SEC" \
  /opt/pi-runtime/bin/node \
  /opt/pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js \
  "${E_SMOKE_SYSTEM_ARGS[@]}" \
  --mode json --provider openai --model qwen35-9b --thinking off \
  --extension /work/config/inference_options.ts \
  --tools read,grep,find,ls --no-approve --no-extensions --no-skills \
  --no-prompt-templates --no-context-files --offline --session-dir /work/sessions \
  -p < "$E_SMOKE_OUTPUT_DIR/prompt.txt" \
  > "$E_SMOKE_OUTPUT_DIR/pi.jsonl" 2> "$E_SMOKE_OUTPUT_DIR/pi.stderr.txt"; then
  E_SMOKE_EXIT_CODE=0
else
  E_SMOKE_EXIT_CODE=$?
fi
printf '%s\n' "$E_SMOKE_EXIT_CODE" > "$E_SMOKE_OUTPUT_DIR/exit-code.txt"
exit "$E_SMOKE_EXIT_CODE"
