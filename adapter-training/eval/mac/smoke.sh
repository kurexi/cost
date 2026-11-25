#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

MODEL=${MODEL:-models/qwen35-08b}
OUT=work/smoke
mkdir -p "$OUT"
start=$(date +%s)

python run.py --backend hf --model "$MODEL" --gold data/eval.jsonl --name smoke-base --limit 7 --batch 7
python grade.py --gold data/eval.jsonl --pred runs/smoke-base/predictions.jsonl --out runs/smoke-base/grade.json > /dev/null

python sft_data.py --model "$MODEL" --gold data/train.jsonl --out "$OUT/sft.jsonl" --limit 8
python ../../agent/eval/train_sft.py --model "$MODEL" --data "$OUT/sft.jsonl" --out "$OUT/adapter" --accum 2 --max-steps 2 --save-every 0

python run.py --backend hf --model "$MODEL" --adapter "$OUT/adapter" --gold data/eval.jsonl --name smoke-sft --limit 7 --batch 7
python grade.py --gold data/eval.jsonl --pred runs/smoke-sft/predictions.jsonl --out runs/smoke-sft/grade.json > /dev/null

python - <<'EOF'
import json
for name in ("smoke-base", "smoke-sft"):
    m = json.load(open(f"runs/{name}/meta.json"))
    o = json.load(open(f"runs/{name}/grade.json"))["summary"]["overall"]
    print(f"{name}: {m['samples']} samples, {m['seconds_per_sample']} s/sample, "
          f"json_valid {o['json_valid']:.2f}, record_core {o['record_core']:.2f}, "
          f"tool calls/sample {o['mean_tool_calls']:.1f}")
EOF
echo "smoke test passed in $(( $(date +%s) - start )) s. Next: ./mac/train.sh"
