#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_BONSAI=0
[[ "${1:-}" == "--with-bonsai" ]] && WITH_BONSAI=1

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "warning: this is meant for an Apple Silicon Mac (found $(uname -s) $(uname -m)); continuing anyway" >&2
fi

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed"' \
  || { echo "Install a newer Python, e.g. 'brew install python@3.12', then PYTHON=python3.12 ./mac/setup.sh" >&2; exit 1; }

if [[ ! -d .venv ]]; then
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch "transformers>=5" peft accelerate safetensors huggingface_hub

python - <<'EOF'
import torch
print("torch", torch.__version__, "| mps available:", torch.backends.mps.is_available())
if not torch.backends.mps.is_available():
    print("warning: the Mac GPU (mps) is not available to torch; everything will run on CPU")
EOF

mkdir -p models
python - "$WITH_BONSAI" <<'EOF'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="models/qwen35-08b")
if sys.argv[1] == "1":
    snapshot_download("Qwen/Qwen3.5-2B", local_dir="models/qwen35-2b")
    hf_hub_download("prism-ml/Ternary-Bonsai-2-2B-gguf", "Ternary-Bonsai-2-2B-Q2_0.gguf", local_dir="models/bonsai-gguf")
print("models downloaded")
EOF

if [[ "$WITH_BONSAI" == "1" && ! -f models/bonsai-2b/model.safetensors ]]; then
  python bonsai_unpack.py --gguf models/bonsai-gguf/Ternary-Bonsai-2-2B-Q2_0.gguf \
    --base models/qwen35-2b --out models/bonsai-2b --verify
fi

if [[ ! -f data/train.jsonl ]]; then
  python make_eval_set.py --split train --per-locale 300 --users-per-locale 4 --seed 1
fi

python -m unittest discover -s tests -q
echo
echo "Setup done. Next: ./mac/smoke.sh   (a few minutes; checks the whole pipeline end to end)"
