#!/usr/bin/env bash
set -euo pipefail
export QA_AUDIT_CANDIDATE_K=20
export QA_AUDIT_SEMANTIC=1
export QA_SEMANTIC_MODEL=Qwen/Qwen3.5-9B-Base
export QA_SEMANTIC_LOCAL_ONLY=1
export QA_SEMANTIC_BATCH_SIZE=8
export QA_SEMANTIC_MAX_LENGTH=256
export QA_SEMANTIC_QUERY_PREFIX="query: "
export QA_SEMANTIC_PASSAGE_PREFIX="passage: "
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
