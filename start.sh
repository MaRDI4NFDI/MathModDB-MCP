#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set in the current environment." >&2
  exit 1
fi

run_envs=()
for var in OPENAI_API_KEY OPENAI_BASE_URL OPENAI_ORG_ID OPENAI_PROJECT; do
  if [[ -n "${!var:-}" ]]; then
    run_envs+=(-e "$var")
  fi
done

DOCKER_BUILDKIT=1 docker build \
  --secret id=OPENAI_API_KEY,env=OPENAI_API_KEY \
  -t mathmoddb-mcp \
  .

docker run \
  --rm \
  --name mathmoddb-mcp \
  -p 8000:8000 \
  "${run_envs[@]}" \
  mathmoddb-mcp
