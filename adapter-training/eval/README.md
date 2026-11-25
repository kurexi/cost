# Receipt-agent evaluation and training kit

Everything needed to measure, and train, a small open model (Qwen3.5-0.8B
here) as the app's extraction agent for devices without Apple Intelligence.
It reuses the app's own contract: the same instructions, the same prompt
assembly, the same JSON schema, the same three tools and the same tool
budget. That way a score here means what it would mean in the app.

```
make_eval_set.py   multilingual receipts + gold records + simulated users (ledgers), from
                   the agent's generator (agent/eval/receipts.py, agent/eval/locales.py)
env.py             the environment: calculator / searchReceipt / searchRecords,
                   AgentToolBudget, schema + tool descriptions read from the Swift
constrained.py     the app's json_schema response format as a transformers LogitsProcessor
run.py             runs any model on the set the way LocalModelClient does; writes
                   predictions + full transcripts (the "reasoning history")
grade.py           field-level grading against gold, breakdowns, and the RL reward
compare.py         runs side by side on their shared samples: deltas, breakdowns, what changed
sft_data.py        target trajectories from the gold labels
../../agent/eval/train_sft.py  LoRA SFT (completion-only loss, --loss sequence (default) or token mean)
train_grpo.py      GRPO over full tool-using episodes, verifiable reward
tests/             pins env.py and grade.py to the app's behaviour
RESULTS.md         scores so far: the untuned quantized baselines and the 16-bit reference; add each training round
```

Stdlib only except where a model runs (`run.py --backend hf`, `sft_data.py`,
`train_*.py` need `torch`, `transformers>=5`, `peft`).

## The data

```sh
python3 make_eval_set.py --split eval  --per-locale 40  --seed 7      # 280 samples, 14 users
python3 make_eval_set.py --split train --per-locale 300 --users-per-locale 4 --seed 1   # 2100, 28 users
```

Samples come from seven home locales: en-GB, en-US, zh, ja, fr, es and de,
plus es-MX as a travel destination. Each sample is one of:
- a till receipt, as Vision reads it: column splits, light OCR damage on
  letters (never on digits)
- an order email
- a bank notification (the date often has no year)
- a wallet screenshot (the card is the heading)
- a casual note ("昨天在星巴克花了35块")
- a non-spending message (refunds, transfers received, one-time codes,
  deliveries)

Each sample belongs to a simulated user. Their categories carry the
user-language titles (超市, 食料品, Lebensmittel…), and their ledger is
what `searchRecords` searches. The prompt's category and past-purchase
lines are derived from that ledger the way `CategoryFit` derives them from
the store.

`gold` is the record the content should produce, plus what fair grading
needs:
- every accepted merchant spelling
- whether the payment instrument is graded exactly, must be empty, or is
  free (cash, a bare card scheme)
- acceptable titles for a proposed category
- the items, for the note

`traits` say what makes the sample hard, and results break down by them:
- the amount: `no_printed_total`, `cash_change`, `was_price`, `tax_added`,
  `tip`, `shipping_added`, `column_split`, `ocr_noise`
- merchant vs. instrument: `card_heading`, `issuer_on_receipt`
- currency and dates: `travel`, `date_without_year`, `relative_date`
- category: `needs_proposal`, `mixed_merchant`, `ledger_lookup_useful`

Instrument names are partly invented per seed, as in `make_training_data.py`,
so train and eval never share them.

## Running a model and grading it

```sh
# any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, vLLM): uses the app's json_schema
python3 run.py --backend openai --base-url http://localhost:11434/v1 --model qwen3:8b \
    --gold data/eval.jsonl --name qwen3-8b
# a local Hugging Face checkpoint (+ optional LoRA adapter)
python3 run.py --backend hf --model /path/Qwen3.5-0.8B [--adapter adapters/sft] \
    --gold data/eval.jsonl --name sft --limit 70 --batch 10
python3 grade.py --gold data/eval.jsonl --pred runs/sft/predictions.jsonl --out runs/sft/grade.json
# against a committed quantized baseline (untuned Bonsai 4B; bonsai-2b for the 0.5 GiB budget),
# on the same 70 samples as --limit 70. A 16-bit run like this one isn't the final score:
# quantize the tuned model and evaluate that file (see RESULTS.md)
python3 compare.py bonsai-4b sft
```

`run.py` grades its own run at the end (`runs/<name>/grade.json`), so the
`grade.py` step is only needed for predictions made elsewhere.

`runs/<name>/transcripts.jsonl` keeps every turn: the tool calls, what the
tools answered, and the raw generations. To distill reasoning from a stronger
local model, run it here, keep the transcripts whose grade is `record_all`,
and train on them in place of (or beside) `sft_data.py`'s templated targets.

## Training

```sh
python3 sft_data.py  --model $M --gold data/train.jsonl --out sft.jsonl --limit 240
python3 ../../agent/eval/train_sft.py --model $M --data sft.jsonl --out adapters/sft --accum 8 --lr 2e-4
python3 train_grpo.py --model $M --init-adapter adapters/sft --gold data/train.jsonl \
    --out adapters/grpo --steps 200 --prompts 4 --group 8 --skip 1000
```

The reward (`grade.reward`) is graded against the gold labels, per field:
amount 0.30, merchant 0.15, category 0.15, spending 0.10, currency 0.10,
instrument 0.08, date 0.08, time 0.04. On top of that:
- -1 for an answer the app couldn't decode
- -0.03 for each wasted tool call
- +0.05 for a fully correct record

The app's own audits (`totalIsPlausible`, merchant grounding) are kept out of
the reward deliberately. They are necessary but not sufficient, and a policy
rewarded on them learns to copy any plausible number.

## Keeping it in sync

`env.py` is a reference port of Swift that is the source of truth. It reads
the schema and tool descriptions from `LocalModelAgent.swift` and
`LedgerSearch.swift` directly, and `tests/` mirrors the Swift tests. If you
change a tool's behaviour or a message in the app, change `env.py` and run
`python3 -m unittest discover -s tests`.
