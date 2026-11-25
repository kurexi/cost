# Using the agent

The ledger is a folder of CSV files; the model is any OpenAI-compatible
server (llama-server, Ollama, LM Studio, vLLM). Standard library only, no
install step. Run everything from the repository root.

The trained model is on Hugging Face as
[kurexi/cost-0.8b-q8](https://huggingface.co/kurexi/cost-0.8b-q8), a single
0.81 GB GGUF (`qwen3.5-0.8b-grpo-r2-q8_0.gguf`). llama-server can fetch and
serve it in one step; `--jinja` is needed for tool calls:

```bash
llama-server -hf kurexi/cost-0.8b-q8 --alias cost --jinja --port 8090
```

For the smaller 4-bit file (0.53 GB), use
[kurexi/cost-0.8b-q4_k_m](https://huggingface.co/kurexi/cost-0.8b-q4_k_m)
instead:

```bash
llama-server -hf kurexi/cost-0.8b-q4_k_m --alias cost --jinja --port 8090
```

Start a ledger with your categories and tags:

```bash
python3 -m spending_agent init ~/ledger --category Groceries --category "Eating Out" --tag Work
```

Add a record by hand:

```bash
python3 -m spending_agent add ~/ledger --date 2026-11-01 --merchant Tesco --amount 12.40 --currency GBP --category Groceries
```

Let the agent read a receipt, here against a server on port 8090:

```bash
python3 -m spending_agent process ~/ledger receipt.txt --base-url http://127.0.0.1:8090/v1 --model cost --reasoning-effort none
```

See what is waiting for review:

```bash
python3 -m spending_agent review ~/ledger
```

Read the agent's memory and its history:

```bash
python3 -m spending_agent memory ~/ledger --history
```

Each `process` run writes one row to `review.csv` and the whole session to
`sessions/<id>.json` (messages, tool calls, findings, the contract
fingerprint). Every draft goes to review. If the server needs a key, set
`SPENDING_AGENT_API_KEY`.

Tests:

```bash
python3 -m unittest discover -s tests -t .
```

Scoring a model of your own, and training one: [eval/README.md](eval/README.md).
