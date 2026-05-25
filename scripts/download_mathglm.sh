#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${MATHGLM_CHECKPOINT_DIR:-$ROOT/checkpoints/mathglm-2b}"
BASE="https://cloud.tsinghua.edu.cn/d/cf429216289948d889a6/files"

mkdir -p "$DEST/1"

wget -O "$DEST/model_config.json" "${BASE}/?p=%2Fmodel_config.json&dl=1"
wget -O "$DEST/latest" "${BASE}/?p=%2Flatest&dl=1"
wget -O "$DEST/1/mp_rank_00_model_states.pt" \
  "${BASE}/?p=%2F1%2Fmp_rank_00_model_states.pt&dl=1"

echo "MathGLM-2B weights saved to $DEST"
