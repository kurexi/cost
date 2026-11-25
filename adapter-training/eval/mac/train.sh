#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

MODEL=${MODEL:-models/qwen35-08b}
TARGET=${TARGET:-0.95}
OUT=${OUT:-work/$(basename "$MODEL")}
mkdir -p "$OUT"

python iterate.py --model "$MODEL" --out "$OUT" --target "$TARGET" --batch "${BATCH:-16}" 2>&1 | tee -a "$OUT.log"
echo
echo "Per-round scores: $OUT/rounds.jsonl   Full log: $OUT.log"
echo "Per-language / per-trait breakdown of the last round: runs/iter_r*/grade.json"
