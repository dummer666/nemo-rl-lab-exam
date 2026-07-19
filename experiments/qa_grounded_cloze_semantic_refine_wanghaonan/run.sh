#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export CLOZE_CRITIC_POOL_TRAIN=800
export CLOZE_CRITIC_POOL_VALIDATION=120
export CLOZE_CRITIC_REUSE_DIR="/shared/outputs/wanghaonan/qa_grounded_cloze_semantic_wanghaonan/qa_grounded_cloze_semantic_wanghaonan-wanghaonan-20260719-065000/grounded_cloze_semantic"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
