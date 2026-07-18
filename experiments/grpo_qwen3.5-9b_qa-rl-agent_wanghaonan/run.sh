#!/usr/bin/env bash
# Multi-turn QA retrieval GRPO. The shared launcher auto-selects this directory's run.py.
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
