import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_EVAL = next(folder for folder in (os.path.abspath(os.path.join(HERE, "..", "..", "agent", "eval")),
                                        os.path.abspath(os.path.join(HERE, "..", "..", "eval")))
                  if os.path.exists(os.path.join(folder, "receipts.py")))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.append(AGENT_EVAL)

import make_training_data as mtd
from receipts import ZERO_DECIMAL, fold, gold_amount_string

INSTRUCTIONS = mtd.INSTRUCTIONS
assemble_prompt = mtd.assemble_prompt

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
