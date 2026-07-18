#!/usr/bin/env bash
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENTRY="${ENTRY:-${EXP_DIR}/run.py}"

exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
