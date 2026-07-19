#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export CLOZE_CRITIC_POOL_TRAIN=24
export CLOZE_CRITIC_POOL_VALIDATION=12
export CLOZE_CRITIC_MAX_PER_SPLIT=8
export CLOZE_CRITIC_BATCH_SIZE=4
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
