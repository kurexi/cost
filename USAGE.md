# Using the agent

The ledger is a folder of CSV files; the model is any OpenAI-compatible
server (llama-server, Ollama, LM Studio, vLLM). Standard library only, no
install step. Run everything from the repository root.

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
python3 -m spending_agent process ~/ledger receipt.txt --base-url http://127.0.0.1:8090/v1 --model bonsai --reasoning-effort none
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
