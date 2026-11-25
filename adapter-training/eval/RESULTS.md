# Results

Scores on `data/eval.jsonl` (seed 7), graded by `grade.py`. Columns are the
fraction of samples with that field exactly right; `record_core` is the
spending verdict, amount, currency, merchant and category all right at once,
and `record_all` adds the instrument, date and time. The target for a model
the app can rely on is `record_core` ≥ 0.95.

All runs so far use `--limit 70`: the first 10 samples of each home locale,
the same 70 every time, so rows compare directly. At n = 70 a difference
of about ±0.1 overall is noise, and a single sample moves a language or
trait row by 0.1–0.3. Read the small rows as pointers, not measurements.

**Every run below used the earlier past-records tool.** Up to the 4B and
MLX runs, the third tool was `searchPastPurchases` (one `query`, a list of
matching purchases) with instructions that described it. `main` replaced it
with `searchRecords` (a word and/or an exact amount, answered with the
matching merchants, their categories, counts and how regularly they recur),
rewrote that part of the instructions, and ends the tool phase after a
second refused call. The kit now mirrors `main`. Runs made from now on are
on the new tool, so a difference against the rows below is partly the tool
and the instructions, not only the model. Rerun a baseline before reading a
small difference.

## What counts as the baseline

The goal is a model that is **quantized, small in memory, and accurate
after tuning**. So the baseline is an untuned model that is already
quantized, at the memory the tuned model will have:

| memory budget | baseline (untuned, quantized) | record_core |
|---|---|---|
| ≈ 0.5 GiB | Ternary-Bonsai-2-2B, `PQ2_0`, 0.48 GiB | 0.36 |
| ≈ 1 GiB | Ternary-Bonsai-2-4B, `PQ2_0`, 1.0–1.1 GiB | 0.51 |

The 16-bit Qwen3.5-0.8B (section 1) is **not** a baseline. It is an
unquantized reference: 1.40 GiB as run, and nobody would ship it that way.
Its score says what the architecture does at full precision, not what a
shippable model does. A fair 0.8B baseline is a quantized 0.8B (a 4-bit
GGUF is about 0.5 GiB, the 2B's budget) run through `llama-server` like the
Bonsai models. The one attempt so far, MLX 4-bit on a Mac (section 5),
scored 0.00 and probably measures the server, not the model, so **a valid
quantized 0.8B score doesn't exist yet.**

The same rule applies after tuning: **a tuned model's score is the score of
its quantized file**, measured through the runtime that will serve it. Two
things follow.

- Training runs on 16-bit weights (the 0.8B checkpoint, or Bonsai unpacked
  by `bonsai_unpack.py`). A LoRA merged into unpacked Bonsai weights leaves
  them no longer ternary, so the merged model has to be quantized again, and
  quantizing costs accuracy: little at 4 bits, a lot at 2 bits unless the
  training itself is quantization-aware. Nothing here measures that loss yet.
- So each training round needs two numbers: the 16-bit score `iterate.py`
  reports, and the score of the quantized export. Only the second is
  comparable with the table above.

## Summary

All untuned, same 70 samples, same tools and limits, answers constrained to
the app's JSON schema. "Right when decodable" is `record_core ÷ json_valid`,
the share right among answers the app could read. An undecodable answer
fails every field, so this separates getting the format wrong from getting
the record wrong.

This is the only place the models are compared. The sections below hold
each run's own details and breakdown.

| model | weights | weight memory | s/sample | record_core | record_all | json_valid | right when decodable |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 16-bit, unquantized | 1.40 GiB | 10.1 (batch 16) | 0.34 | 0.04 | 0.89 | 0.38 |
| Ternary-Bonsai-2-2B | ternary PQ2_0, 2.13 bits | 0.48 GiB | 17.0 | 0.36 | 0.13 | 0.70 | 0.51 |
| Ternary-Bonsai-2-4B | ternary PQ2_0, 2.13 bits | 1.0–1.1 GiB | 37.8 | 0.51 | 0.24 | 0.81 | 0.63 |
| Ternary-Bonsai-2-27B | ternary PQ2_0, 2.13 bits | 6.70 GiB | 40.8 | **0.79** | **0.67** | 0.99 | 0.80 |

Compare the models by **weight memory**, not by parameter count. The 2B
ties the 0.8B with a third of its memory. The 4B beats the 0.8B (0.51
against 0.34) with about three-quarters of its memory (1.0–1.1 GiB against 1.40). The 27B is far ahead but
needs nearly 5 times the 0.8B's memory. Across the three Bonsai sizes,
record_core rises steadily with size: 0.36, 0.51, 0.79. No model between
the 4B and the 27B has been run, so where in that gap the curve bends is
unknown. The machines and serving differ too: the
0.8B ran 16 samples per batch on a T4, the Bonsai models one request at a time, as
the app sends them (the 2B and 4B on the same Steam Deck, the 27B on a T4).
So the s/sample column compares setups, not models; only the 2B and 4B
share one, where the 4B takes 2.2 times as long. The
runs themselves are in `adapter-training/eval.ipynb`.

**Is it more than noise?** Against the 0.8B reference (24 of 70 records right), the
27B (55) is: across every split of fixed and broken samples its totals
allow, the exact sign test gives p ≤ 0.0001. The 4B (36) isn't settled by
totals alone: p is anywhere from 0.0005 to 0.16 depending on how many
samples flipped each way, and only the per-sample predictions decide it.
The 2B (25) is one sample off the 0.8B.

### Every field

| | record_core | record_all | amount | currency | merchant | category | instrument | date | json_valid |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (T4) | 0.34 | 0.04 | 0.73 | 0.70 | 0.64 | 0.39 | 0.60 | 0.29 | 0.89 |
| Bonsai 2B | 0.36 | 0.13 | 0.53 | 0.64 | 0.61 | 0.50 | 0.53 | 0.20 | 0.70 |
| Bonsai 4B | 0.51 | 0.24 | 0.61 | 0.73 | 0.74 | 0.69 | 0.76 | 0.50 | 0.81 |
| Bonsai 27B | **0.79** | **0.67** | 0.94 | 0.94 | 0.97 | 0.84 | 0.90 | 0.89 | 0.99 |

The 4B is better than the 2B on every column. An undecodable answer zeroes
every field, so the 2B's amount, merchant and instrument fall below the
0.8B's on format alone (json_valid 0.70), while its category is better.

### The hard cases

The rows the 0.8B reference fails (section 1), for every model:

| row | 0.8B | 2B | 4B | 27B |
|---|---|---|---|---|
| `needs_proposal`: category | 0.00 | 0.00 | 0.40 | 0.60 |
| `non_spending`: record_core | 0.22 | 0.78 | 0.78 | 0.89 |
| `travel`: currency | 0.33 | 0.17 | 0.33 | 0.83 |
| `date_without_year`: date | 0.12 | 0.00 | 0.62 | 1.00 |
| `relative_date`: date | 0.17 | 0.00 | 0.50 | 1.00 |
| `email`: instrument | 0.00 | 0.00 | 0.17 | 1.00 |
| `cash_change`: json_valid | 0.57 | 0.57 | 0.71 | 1.00 |
| `wallet`: amount | 0.70 | 0.10 | 0.30 | 0.80 |
| `casual`: record_core | 0.33 | 0.17 | 0.33 | 0.33 |

Only the 27B handles foreign receipts. Casual notes stay at 0.33 even for
the 27B, which points at the answer key as much as the models (see
section 2, *Where it still fails*).

### Model details

| | Qwen3.5-0.8B | Ternary-Bonsai-2-2B | Ternary-Bonsai-2-4B | Ternary-Bonsai-2-27B |
|---|---|---|---|---|
| architecture | Qwen3.5 (hybrid: Gated DeltaNet + full attention) | Qwen3.5-2B | Qwen3.5-4B | Qwen3.5-27B |
| parameters | 752 M in the language model (the checkpoint's 873 M adds a 101 M vision tower and a 20 M multi-token-prediction head, neither used here) | 1.88 B | ≈ 4.0–4.4 B (implied by the file size at 2.13 bits) | 26.90 B |
| weight format | the published bf16 checkpoint, unquantized | ternary, PrismML's `PQ2_0`, group 128: each weight is −1, 0 or +1 times its group's scale | `PQ2_0`, as the 2B | `PQ2_0`, as the 2B |
| stored bits per weight | 16 | 2.13: 128 weights × 2-bit codes + a 16-bit scale = 34 bytes per group | 2.13 | 2.13 (as `llama-bench` reports the format) |
| weight memory | 1.40 GiB (752 M × 2 bytes) | 0.48 GiB (the file: `ls -lh` shows 494M; the quantized weights alone compute to 0.47 GiB, the rest is the tokenizer and other metadata) | 1.0–1.1 GiB (the file: `ls -lh` shows 1.1G, which rounds up) | 6.70 GiB (measured by `llama-bench`) |
| precision at run time | CPU run: bf16. T4 run: fp16 (`eval.ipynb`, cell 3) | weights dequantized by the fork's kernels; KV cache at `llama-server`'s default f16 | same as the 2B | same as the 2B |
| file | Qwen/Qwen3.5-0.8B (Hugging Face) | `bonsai2-gguf/2B/Ternary-Bonsai-2-2B-PQ2_0.gguf` (Bonsai demo) | `bonsai2-gguf/4B/Ternary-Bonsai-2-4B-PQ2_0.gguf` (Bonsai demo) | `bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` (Bonsai demo) |
| runtime | transformers, schema enforced by `constrained.py` | `llama-server` on a Steam Deck, schema as `response_format` | `llama-server` on the Steam Deck, as the 2B | Bonsai demo's prebuilt CUDA `llama-server` (build b10828), Colab T4 |

"Ternary" is about 1.58 bits of information per weight, stored in 2-bit
codes. The bits above are what the file stores, including the scales.

The three Bonsai models ran in the same format (`PQ2_0`, group 128) from the
same demo package, so the differences between them are model size alone.
The 2B's parameter count is from the same model's `Q2_0` release, read from
its header (1.88 B, with the embedding and the tied output head quantized
too).

**The 4B's size** is from its file on the Deck (`ls -lh`: 1.1G, which
GNU `ls` rounds up, so 1.0–1.1 GiB). The parameter count is implied from
that at 2.13 bits per weight, not read from the header.

**Not recorded for the 2B and 4B runs:** the llama.cpp build and whether
they ran on the Deck's CPU or GPU. The server's start-up log prints both.

## 1. Untuned Qwen3.5-0.8B, 16-bit (unquantized reference)

Tools on (calculator, searchReceipt, and the earlier searchPastPurchases), `max_rounds` 4,
`tool_tokens` 160, `answer_tokens` 600, greedy decoding, the app's JSON
schema enforced on the answer.

Qwen3.5-0.8B as published: bf16 weights, 752 M parameters in the language
model, 1.40 GiB, no quantization.

| run | where | dtype | s/sample |
|---|---|---|---|
| `runs/base-tools` | CPU, this repo's dev container | bfloat16 | 69 |
| T4 | Colab T4 via the VS Code extension, `eval.ipynb`: `flash-linear-attention` 0.6.0 kernels for the delta rule, reference PyTorch for `causal_conv1d`; batch 16 | float16 | 10.1 (70 in 11.8 min, ~48 tok/s) |

### Overall

| | record_core | record_all | amount | currency | merchant | category | instrument | date | json_valid |
|---|---|---|---|---|---|---|---|---|---|
| CPU | 0.37 | 0.10 | 0.77 | 0.70 | 0.63 | 0.44 | 0.61 | 0.33 | 0.89 |
| T4  | 0.34 | 0.04 | 0.73 | 0.70 | 0.64 | 0.39 | 0.60 | 0.29 | 0.89 |

The two agree within noise. Greedy decoding diverges between bf16 on a CPU
and fp16 on a GPU, so individual samples differ, but the Colab pipeline
reproduces the CPU run.

### T4 run, full breakdown

```
                          n record_co record_al    amount  currency  merchant  category instrumen      date json_vali
overall                  70      0.34      0.04      0.73      0.70      0.64      0.39      0.60      0.29      0.89
-- language
de                       13      0.31      0.00      0.77      0.92      0.62      0.31      0.77      0.31      1.00
en                       22      0.32      0.00      0.59      0.55      0.64      0.36      0.41      0.23      0.86
es                        8      0.75      0.12      1.00      1.00      1.00      0.75      0.75      0.25      1.00
fr                        8      0.12      0.00      0.75      0.50      0.50      0.25      0.62      0.00      0.75
ja                       10      0.10      0.10      0.70      0.60      0.60      0.20      0.70      0.50      0.90
zh                        9      0.56      0.11      0.78      0.78      0.56      0.56      0.56      0.44      0.78
-- style
casual                   12      0.33      0.00      1.00      0.75      0.92      0.42      1.00      0.08      1.00
email                     6      0.50      0.00      0.67      0.67      0.67      0.50      0.00      0.17      0.67
non_spending              9      0.22      0.22      0.22      0.22      0.22      0.22      0.22      0.22      0.89
notification              8      0.38      0.00      1.00      0.88      0.75      0.38      0.62      0.12      1.00
till                     25      0.32      0.04      0.72      0.76      0.56      0.36      0.72      0.44      0.80
wallet                   10      0.40      0.00      0.70      0.80      0.80      0.50      0.50      0.40      1.00
-- trait
card_heading             10      0.40      0.00      0.70      0.80      0.80      0.50      0.50      0.40      1.00
cash_change               7      0.14      0.00      0.57      0.43      0.57      0.29      0.57      0.29      0.57
column_split              7      0.43      0.14      0.86      1.00      0.57      0.43      0.86      0.71      1.00
date_without_year         8      0.38      0.00      1.00      0.88      0.75      0.38      0.62      0.12      1.00
issuer_on_receipt         9      0.33      0.00      0.78      0.89      0.44      0.33      0.67      0.44      0.89
mixed_merchant            4      0.25      0.25      1.00      1.00      0.75      0.25      1.00      0.50      1.00
needs_proposal           10      0.00      0.00      0.80      0.80      0.40      0.00      0.50      0.30      0.90
no_printed_total          1      0.00      0.00      0.00      1.00      0.00      0.00      1.00      1.00      1.00
non_spending              9      0.22      0.22      0.22      0.22      0.22      0.22      0.22      0.22      0.89
ocr_noise                 6      0.50      0.00      0.67      1.00      0.67      0.50      1.00      0.67      1.00
relative_date             6      0.33      0.00      1.00      0.50      1.00      0.50      1.00      0.17      1.00
secondary_kind            2      0.00      0.00      1.00      1.00      0.50      0.00      1.00      0.50      1.00
shipping_added            6      0.50      0.00      0.67      0.67      0.67      0.50      0.00      0.17      0.67
subtotal_line             6      0.33      0.17      0.67      0.67      0.67      0.50      0.83      0.67      0.83
tax_added                 2      1.00      0.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00
travel                   12      0.00      0.00      0.67      0.33      0.75      0.17      0.75      0.25      0.92
was_price                 3      0.33      0.00      1.00      1.00      0.67      0.33      0.67      0.33      1.00
```

`language` is the language of the content, not the user's home locale, so
the counts are uneven (an en-GB user's receipt from a Mexico trip counts
under `es`).

### What it can't do

Present in both runs, so these are the model's, not the hardware's:

- **It never proposes a category.** `needs_proposal` category is 0.00 on
  both runs. When none of the user's categories fits, it forces the nearest
  one, which is the failure the two-slot design (`category` /
  `proposedCategory`) exists to prevent.
- **It records money that wasn't spent.** `non_spending` is 0.22 on the T4
  and 0.44 on the CPU: refunds, transfers received, one-time codes and
  delivery notices are drafted as purchases most of the time.
- **It assumes the home currency abroad.** `travel` currency is 0.33 on the
  T4 and 0.17 on the CPU, and `record_core` is 0.00 on both.
- **It can't complete a date.** Date overall is 0.29–0.33. Dates printed
  without a year score 0.12, and relative dates ("yesterday") 0.17. The
  prompt carries today's date, but the model doesn't use it.
- **Emails lose the card.** On `email`, instrument is 0.00 on both runs.
- **About 1 answer in 9 isn't valid JSON, despite the schema.** It's 0.89 on
  both runs and concentrated in cash receipts (`cash_change` 0.57 on both)
  and emails. The likely cause is the answer running into the 600-token
  limit with a long `reasoning` field. Confirm by checking whether the
  invalid answers used all 600 tokens; if so, it's the model's verbosity,
  not a grammar bug.

Already partly there: amount 0.73–0.77, merchant 0.63–0.64, and most
non-travel currencies.

### What it would mean in the app

From the CPU run, approximating the app's gates (the total audit, merchant
grounding, the 0.8 confidence bar, an existing category): 50 of the 70
drafts would have been auto-approved, and only 44% of those were correct.
An untuned 0.8B model must not write to the ledger unattended.

## Comparing a new run

`runs/base-tools` (the 0.8B reference on CPU), `runs/bonsai-2b` and
`runs/bonsai-4b` (the quantized baselines) are committed, so any new run on
the same samples compares against them directly:

```sh
python3 run.py ... --limit 70 --name <new>     # grades itself at the end
python3 compare.py bonsai-4b <new>             # first argument = baseline
```

`compare.py` scores only the samples both runs answered. It prints the
headline metrics with their deltas, one metric by language, style and trait,
and the samples that went from wrong to right and back. The sign test on
those flips says whether the difference is more than noise, which at n = 70
a few points usually isn't.

## 2. Ternary-Bonsai-2-27B (PQ2_0), untuned

PrismML's ternary 27B (weights and format in *Model details*), served by
the fork's prebuilt CUDA `llama-server` from the Bonsai demo, on a
Colab T4, through `run.py --backend openai --no-think` (`eval.ipynb`, cell 5). Same 70 samples as
section 1, same tools, `max_rounds` 4, `tool_tokens` 160, `answer_tokens` 600.
The answer went through the server's `json_schema` response format for every
sample, and no request failed.

| | |
|---|---|
| `llama-bench` on the T4 | prompt 325 tok/s, generation 17.2 tok/s |
| eval speed | 40.8 s per sample, 70 in 47.6 min (one request at a time) |
| tool calls | exactly 4 on all 70 samples, so `max_rounds` cut it off every time |

The per-sample predictions of this run weren't kept (the Colab runtime
restarted), so there is no fixed/broken list. The totals settle significance
on their own: at most 15 samples can have gone from right to wrong against
the 0.8B, the 15 the 27B missed, which bounds p ≤ 0.0001 (see *Summary*).

### Full breakdown

```
                          n record_co record_al    amount  currency  merchant  category instrumen      date json_vali
overall                  70      0.79      0.67      0.94      0.94      0.97      0.84      0.90      0.89      0.99
-- language
de                       13      0.77      0.62      1.00      1.00      1.00      0.77      0.85      0.92      1.00
en                       22      0.77      0.68      0.95      0.86      0.95      0.86      0.91      0.86      1.00
es                        8      0.88      0.62      0.88      1.00      1.00      1.00      0.75      0.88      1.00
fr                        8      0.75      0.62      1.00      1.00      1.00      0.75      1.00      0.88      1.00
ja                       10      0.80      0.70      0.90      0.90      0.90      0.80      0.90      0.80      0.90
zh                        9      0.78      0.78      0.89      1.00      1.00      0.89      1.00      1.00      1.00
-- style
casual                   12      0.33      0.17      0.92      0.75      0.92      0.50      0.92      0.50      0.92
email                     6      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00
non_spending              9      0.89      0.89      0.89      0.89      0.89      0.89      0.89      0.89      1.00
notification              8      0.88      0.62      1.00      1.00      1.00      0.88      0.75      1.00      1.00
till                     25      0.88      0.76      1.00      1.00      1.00      0.88      0.96      0.96      1.00
wallet                   10      0.80      0.70      0.80      1.00      1.00      1.00      0.80      1.00      1.00
-- trait
card_heading             10      0.80      0.70      0.80      1.00      1.00      1.00      0.80      1.00      1.00
cash_change               7      1.00      0.86      1.00      1.00      1.00      1.00      1.00      0.86      1.00
column_split              7      0.86      0.71      1.00      1.00      1.00      0.86      0.86      1.00      1.00
date_without_year         8      0.88      0.62      1.00      1.00      1.00      0.88      0.75      1.00      1.00
issuer_on_receipt         9      0.89      0.67      1.00      1.00      1.00      0.89      0.89      1.00      1.00
mixed_merchant            4      0.25      0.25      0.75      0.75      0.75      0.25      0.75      0.50      0.75
needs_proposal           10      0.50      0.40      0.90      1.00      1.00      0.60      0.90      0.90      1.00
no_printed_total          1      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00
non_spending              9      0.89      0.89      0.89      0.89      0.89      0.89      0.89      0.89      1.00
ocr_noise                 6      1.00      0.83      1.00      1.00      1.00      1.00      0.83      1.00      1.00
relative_date             6      0.33      0.33      1.00      0.67      1.00      0.67      1.00      1.00      1.00
secondary_kind            2      0.00      0.00      1.00      1.00      1.00      0.00      1.00      0.50      1.00
shipping_added            6      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00
subtotal_line             6      1.00      0.83      1.00      1.00      1.00      1.00      0.83      1.00      1.00
tax_added                 2      1.00      0.50      1.00      1.00      1.00      1.00      1.00      1.00      1.00
travel                   12      0.75      0.67      1.00      0.83      1.00      0.92      0.92      1.00      1.00
was_price                 3      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00
```

### Where it still fails

Receipts are close to solved: `till` record_core is 0.88, and amount,
currency and merchant are 1.00 there. What's left is mostly the
**category**, plus the **casual notes**.

- **Casual notes: record_core 0.33**, against 0.88 on receipts. 8 of these
  12 samples fail, mainly on category (0.50), plus currency (0.75) and date
  (0.50). Date resolution itself works (`relative_date` date 1.00), so the
  casual date misses are likely notes with no date at all, where the gold is
  empty and the model fills one in.
- **Same shop, two categories (`mixed_merchant`): 0.25**, and
  `secondary_kind` 0/2. When the user files one shop under two categories,
  the model picks the shop's usual category.
- **Proposing a new category (`needs_proposal`): 0.50.** It now proposes
  where the 0.8B never did, but still forces an existing title in about
  4 cases out of 10.

**Some casual failures may be the eval set's fault, not the model's.**
A casual note often carries no evidence for the category its gold
expects. `8.34 at Whole Foods Market` is gold *Coffee* (`secondary_kind`),
but nothing in the note says coffee. Only the ledger's amounts could hint at
it. `Sainsbury's £9.13 lunch` is gold *Groceries*. And the note templates
add words like "lunch" whatever was bought (`Smyths Toys GBP 31.25 yesterday
lunch`, gold *Kids*). Before treating 0.33 as the model's score, check the
failed casual samples in a rerun's transcripts (this run's weren't kept). Where the gold
is unknowable from the content and ledger, the generator
(`render_casual`, the `mixed_merchant` branch of `build_sample`) should
either stop asserting a category it can't support or add the items that
decide it.

### What it means

- **For the app on a Mac:** a 27B behind `llama-server` is a plausible
  engine for the Mac's local-model path. It gets about 4 in 5 records fully
  right untuned, and the app's audits and review gate stand between the
  remainder and the ledger. It isn't at the 0.95 target, and the remaining
  errors are categories, which the total audit doesn't catch.
- **For the phone:** not deployable. 6.7 GiB of weights is beyond an
  iPhone's memory budget for an app.
- **As a teacher for the 0.8B:** this is the model to distill from (see
  the README). Run it on the **train** split, never on `eval.jsonl`, keep
  the transcripts whose grade is `record_all`, and train the 0.8B on them
  beside or instead of `sft_data.py`'s templated targets. They carry real
  tool use and reasoning that the templates only imitate.
- **`max_rounds`:** all 70 samples used all 4 rounds. A run with
  `--max-rounds 10`, the app's limit, would show whether extra tool use
  lifts the category and casual rows.

## 3. Ternary-Bonsai-2-2B (PQ2_0), untuned

PrismML's ternary 2B (Qwen3.5-2B architecture), served by `llama-server`
on a Steam Deck on the local network, through
`run.py --backend openai --no-think`. Same 70 samples and settings as
sections 1 and 2.

```
python run.py --backend openai --base-url http://steamdeck.lan:8080/v1 --model bonsai --no-think \
    --gold data/eval.jsonl --name bonsai-2b --limit 70 --timeout 900
```

70 samples in 1189 s, so **17.0 s per sample**. 2 requests failed; each
counts as an undecodable answer. Model:
`models/bonsai2-gguf/2B/Ternary-Bonsai-2-2B-PQ2_0.gguf` (the server's
`/props`); weights and format in *Model details*.

It ties the 0.8B on `record_core`, losing on the format (30% of its
answers undecodable) and gaining on the substance (right 51% of the time
when readable, against 38%); the figures are in *Summary*.

Two failed requests explain 2 of the 21 undecodable answers. The other 19
need a look before the format loss is blamed on the model. Each line of the
run's output says which response format the server accepted. If it fell back
from `json_schema` to `json_object` or none, the schema wasn't enforced, and
that's a server or build issue, not the model's. If it's `json_schema`
throughout, the answers likely ran into the 600-token limit, which
`answer_tokens` and the transcripts will show.

### Full breakdown

```
                          n record_co record_al    amount  currency  merchant  category instrumen      date json_vali
overall                  70      0.36      0.13      0.53      0.64      0.61      0.50      0.53      0.20      0.70
-- language
de                       13      0.23      0.00      0.54      0.69      0.62      0.38      0.62      0.08      0.77
en                       22      0.36      0.14      0.55      0.59      0.68      0.45      0.45      0.23      0.73
es                        8      0.38      0.12      0.38      0.62      0.62      0.62      0.38      0.12      0.62
fr                        8      0.12      0.12      0.25      0.38      0.25      0.25      0.25      0.25      0.38
ja                       10      0.50      0.30      0.70      0.80      0.60      0.70      0.80      0.40      0.80
zh                        9      0.56      0.11      0.67      0.78      0.78      0.67      0.67      0.11      0.78
-- style
casual                   12      0.17      0.00      0.25      0.42      0.58      0.25      0.58      0.00      0.58
email                     6      0.67      0.00      0.83      0.83      0.83      0.67      0.00      0.00      0.83
non_spending              9      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.78
notification              8      0.25      0.00      0.62      0.50      0.62      0.38      0.25      0.00      0.62
till                     25      0.40      0.08      0.64      0.84      0.60      0.60      0.72      0.28      0.84
wallet                   10      0.00      0.00      0.10      0.30      0.40      0.30      0.30      0.00      0.40
-- trait
card_heading             10      0.00      0.00      0.10      0.30      0.40      0.30      0.30      0.00      0.40
cash_change               7      0.29      0.14      0.43      0.57      0.43      0.43      0.57      0.29      0.57
column_split              7      0.43      0.14      0.71      1.00      0.71      0.71      0.71      0.29      1.00
date_without_year         8      0.25      0.00      0.62      0.50      0.62      0.38      0.25      0.00      0.62
issuer_on_receipt         9      0.56      0.00      0.89      1.00      0.78      0.67      0.67      0.22      1.00
mixed_merchant            4      0.50      0.25      0.75      0.75      0.75      0.50      0.50      0.25      0.75
needs_proposal           10      0.00      0.00      0.70      0.70      0.50      0.00      0.70      0.00      0.80
no_printed_total          1      0.00      0.00      0.00      1.00      0.00      0.00      1.00      1.00      1.00
non_spending              9      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.78
ocr_noise                 6      0.33      0.17      0.50      1.00      0.67      0.67      0.83      0.50      1.00
relative_date             6      0.17      0.00      0.33      0.33      0.50      0.17      0.50      0.00      0.50
secondary_kind            2      0.00      0.00      0.50      0.50      0.50      0.00      0.00      0.00      0.50
shipping_added            6      0.67      0.00      0.83      0.83      0.83      0.67      0.00      0.00      0.83
subtotal_line             6      0.50      0.17      0.67      0.83      0.67      0.67      0.67      0.33      0.83
tax_added                 2      1.00      0.00      1.00      1.00      1.00      1.00      0.50      0.50      1.00
travel                   12      0.00      0.00      0.33      0.17      0.33      0.08      0.25      0.00      0.42
was_price                 3      0.33      0.33      0.67      1.00      0.67      1.00      0.67      0.67      1.00
```

### Where it fails

- **Wallet screens: record_core 0.00, json_valid 0.40.** 6 of the 10 wallet
  screens produced no readable answer, and none of the other 4 was right.
  The wallet layout, with the card as the heading, is the one this model
  can't handle at all.
- **Travel: 0.00, currency 0.17, json_valid 0.42.** It fails on foreign
  receipts, as the 0.8B did, and more than half its travel answers are
  undecodable.
- **Dates: 0.20 overall, 0.00 on casual notes, notifications and dates
  without a year.** Worse than the 0.8B's 0.29.
- **Never proposes a category (`needs_proposal` 0.00),** like the 0.8B.
- **French content: 0.12, json_valid 0.38.** At n = 8 this is about 5
  undecodable answers. Worth checking whether the fr samples overlap the
  travel and wallet ones before reading it as a language weakness.

Better than the 0.8B: non-spending messages (0.78 vs 0.22) and, among
readable answers, the category.

### What it means

- **Untuned, it isn't usable** on its own, for the same reasons as the 0.8B.
- **Speed: 17 s per receipt on a Steam Deck.** That's a handheld-class APU,
  the closest machine here to phone-class hardware. A receipt in well under
  half a minute is acceptable for a background queue.
- **As a fine-tuning base, it's more promising than its score.** The format
  failures are what SFT fixes first. Training targets the exact JSON, and
  the 0.8B's own validity was 0.89 untuned. Its accuracy on readable answers
  (0.51 against the 0.8B's 0.38) suggests more room once the format is
  learned. Whether it fits a phone is a separate question: about 0.48 GiB of
  ternary weights is plausible, but only a runtime that executes the ternary
  format natively keeps it that small. Unpacked to fp16
  (`bonsai_unpack.py`), it's about 4 GB.

## 4. Ternary-Bonsai-2-4B (PQ2_0), untuned

`models/bonsai2-gguf/4B/Ternary-Bonsai-2-4B-PQ2_0.gguf` (the server's
`/props`; weights and format in *Model details*), served by `llama-server` on the same Steam Deck, with the same command
as the 2B except the run name:

```
python run.py --backend openai --base-url http://steamdeck.lan:8080/v1 --model bonsai --no-think \
    --gold data/eval.jsonl --name bonsai-4b --limit 70 --timeout 900
```

Same 70 samples, tools and limits as sections 1–3. 70 samples in 2643 s,
so **37.8 s per sample**. No request failed, so every undecodable answer is
the model's own.

Its strength is the category (0.69): it's the first of the small models
whose category is better than its amount. Whether its gain over the 0.8B
is more than noise needs the per-sample predictions (see *Summary*).

### Full breakdown

```
                          n record_co record_al    amount  currency  merchant  category instrumen      date json_vali
overall                  70      0.51      0.24      0.61      0.73      0.74      0.69      0.76      0.50      0.81
-- language
de                       13      0.31      0.08      0.46      0.62      0.54      0.46      0.69      0.23      0.69
en                       22      0.59      0.23      0.64      0.68      0.73      0.73      0.68      0.41      0.82
es                        8      0.38      0.12      0.50      0.88      0.88      0.75      0.88      0.50      0.88
fr                        8      0.75      0.25      0.75      0.75      0.88      0.75      0.75      0.50      0.88
ja                       10      0.70      0.60      0.90      0.80      0.80      0.80      0.90      0.80      0.90
zh                        9      0.33      0.22      0.44      0.78      0.78      0.67      0.78      0.78      0.78
-- style
casual                   12      0.33      0.00      0.33      0.50      0.67      0.58      0.67      0.25      0.67
email                     6      0.50      0.00      0.50      0.50      0.50      0.50      0.17      0.33      0.50
non_spending              9      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.89
notification              8      0.38      0.25      0.75      0.75      0.88      0.62      0.88      0.62      0.88
till                     25      0.64      0.32      0.80      0.84      0.72      0.76      0.84      0.64      0.88
wallet                   10      0.30      0.00      0.30      0.80      0.90      0.70      0.90      0.20      0.90
-- trait
card_heading             10      0.30      0.00      0.30      0.80      0.90      0.70      0.90      0.20      0.90
cash_change               7      0.43      0.29      0.71      0.57      0.57      0.71      0.71      0.43      0.71
column_split              7      0.71      0.57      0.86      0.86      0.71      0.71      0.86      0.86      0.86
date_without_year         8      0.38      0.25      0.75      0.75      0.88      0.62      0.88      0.62      0.88
issuer_on_receipt         9      0.78      0.22      0.78      0.89      0.78      0.78      0.78      0.56      0.89
mixed_merchant            4      0.25      0.25      0.50      0.75      0.75      0.25      0.75      0.50      0.75
needs_proposal           10      0.30      0.00      0.60      0.70      0.40      0.40      0.60      0.30      0.70
no_printed_total          1      0.00      0.00      1.00      1.00      0.00      1.00      1.00      1.00      1.00
non_spending              9      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.78      0.89
ocr_noise                 6      0.67      0.50      0.83      0.83      0.67      0.83      0.83      0.83      0.83
relative_date             6      0.33      0.00      0.33      0.50      0.83      0.83      0.83      0.50      0.83
secondary_kind            2      0.00      0.00      0.50      1.00      1.00      0.00      1.00      0.50      1.00
shipping_added            6      0.50      0.00      0.50      0.50      0.50      0.50      0.17      0.33      0.50
subtotal_line             6      0.50      0.33      0.67      0.50      0.67      0.67      0.67      0.50      0.67
tax_added                 2      1.00      0.50      1.00      1.00      1.00      1.00      1.00      1.00      1.00
travel                   12      0.17      0.00      0.33      0.33      0.75      0.50      0.75      0.25      0.75
was_price                 3      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00      1.00
```

### Where it fails

- **Wallet screens: the amount, not the format.** The answers are readable
  (json_valid 0.90) and merchant (0.90) and card (0.90) are right, but the
  amount is right only 3 times in 10 and the date twice. The 2B mostly
  couldn't answer at all on these; the 4B answers with the wrong number.
  Something else on a wallet screen is being read as the amount. Check
  which in the transcripts before training on it.
- **Travel: 0.17, currency 0.33.** Like the 0.8B and the 2B, it assumes the
  home currency abroad. Only the 27B handles foreign receipts.
- **Emails: 0.50, with half the answers unreadable** (json_valid 0.50, all
  of them `shipping_added`). No request failed, so these are answers the
  model wrote. Emails are the longest content, so the answers likely ran
  into the 600-token limit.
- **Dates: 0.50 overall, 0.20 on wallet screens and 0.25 on casual notes.**
  Better than the 2B (0.20) and the 0.8B (0.29), far from the 27B (0.89).
- **New categories (`needs_proposal`): 0.30,** against 0.00 for the 2B and
  the 0.8B and 0.50 for the 27B.
- **By language, de 0.31 and zh 0.33 against ja 0.70 and fr 0.75.** At 8–13
  samples per language, that's 2–4 samples either way. Don't read it as a
  language weakness without a larger run.

### What it means

- **The best small model so far.** Its weak spots, the wallet amount,
  foreign currency, long emails and new categories, are specific enough to
  target in training data.
- **Speed:** still workable for a background queue on the Steam Deck, but
  it buys its accuracy with twice the 2B's time per receipt.

## 5. Qwen3.5-0.8B, MLX 4-bit, untuned: not a valid score yet

`mlx-community/Qwen3.5-0.8B-4bit`, served by `mlx_lm.server` on a Mac and
evaluated through `run.py --backend openai --no-think`. Same 70 samples,
tools and limits as the other runs.

| | |
|---|---|
| weights | MLX 4-bit, group 64, with a 16-bit scale and bias per group: about 4.5 stored bits per weight |
| weight memory | ≈ 0.4–0.5 GiB (752 M × 4.5 bits; not measured from the files) |
| runtime | `mlx_lm.server` on a Mac (model and macOS version not recorded) |
| speed | 70 samples in 249 s, **3.6 s per sample**, 0 failed requests |

```
                          n record_co record_al    amount  currency  merchant  category instrumen      date json_vali
overall                  70      0.00      0.00      0.00      0.00      0.44      0.24      0.36      0.06      0.57
-- language
de                       13      0.00      0.00      0.00      0.00      0.46      0.15      0.46      0.08      0.54
en                       22      0.00      0.00      0.00      0.00      0.45      0.18      0.18      0.00      0.59
es                        8      0.00      0.00      0.00      0.00      0.12      0.25      0.12      0.00      0.25
fr                        8      0.00      0.00      0.00      0.00      0.62      0.38      0.50      0.12      0.62
ja                       10      0.00      0.00      0.00      0.00      0.40      0.20      0.50      0.20      0.50
zh                        9      0.00      0.00      0.00      0.00      0.56      0.44      0.56      0.00      0.89
-- style
casual                   12      0.00      0.00      0.00      0.00      0.50      0.33      0.58      0.25      0.58
email                     6      0.00      0.00      0.00      0.00      0.83      0.50      0.00      0.00      1.00
non_spending              9      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.22
notification              8      0.00      0.00      0.00      0.00      0.25      0.00      0.12      0.00      0.50
till                     25      0.00      0.00      0.00      0.00      0.56      0.32      0.64      0.04      0.64
wallet                   10      0.00      0.00      0.00      0.00      0.40      0.20      0.10      0.00      0.50
-- trait
card_heading             10      0.00      0.00      0.00      0.00      0.40      0.20      0.10      0.00      0.50
cash_change               7      0.00      0.00      0.00      0.00      0.43      0.14      0.43      0.14      0.43
column_split              7      0.00      0.00      0.00      0.00      0.86      0.43      0.86      0.00      0.86
date_without_year         8      0.00      0.00      0.00      0.00      0.25      0.00      0.12      0.00      0.50
issuer_on_receipt         9      0.00      0.00      0.00      0.00      0.67      0.56      0.67      0.00      0.67
mixed_merchant            4      0.00      0.00      0.00      0.00      0.50      0.25      0.75      0.25      0.75
needs_proposal           10      0.00      0.00      0.00      0.00      0.30      0.00      0.20      0.10      0.40
no_printed_total          1      0.00      0.00      0.00      0.00      0.00      0.00      1.00      0.00      1.00
non_spending              9      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.22
ocr_noise                 6      0.00      0.00      0.00      0.00      0.50      0.17      0.67      0.00      0.67
relative_date             6      0.00      0.00      0.00      0.00      0.50      0.50      0.67      0.00      0.67
secondary_kind            2      0.00      0.00      0.00      0.00      0.00      0.00      0.50      0.00      0.50
shipping_added            6      0.00      0.00      0.00      0.00      0.83      0.50      0.00      0.00      1.00
subtotal_line             6      0.00      0.00      0.00      0.00      0.83      0.33      0.83      0.00      0.83
tax_added                 2      0.00      0.00      0.00      0.00      0.50      0.50      0.50      0.00      0.50
travel                   12      0.00      0.00      0.00      0.00      0.58      0.08      0.33      0.00      0.67
was_price                 3      0.00      0.00      0.00      0.00      0.33      0.33      0.67      0.00      0.67
```

### Why this isn't the model's score

**Amount and currency are 0.00 on every row, including rows where every
answer was readable** (`email`, `shipping_added`: json_valid 1.00). Merchant
is right on 0.44 of samples, so the answers are JSON with the right
`merchant` key but no correct `totalAmount` or `currencyCode` anywhere. A
weak model gets some amounts right by chance; the 16-bit 0.8B got 0.73.
Zero across 70 samples is what you'd see if the answers don't follow the
app's schema at all, for example `"amount"` or `"total"` instead of
`totalAmount`. The grader reads only the schema's field names, so it scores
those as missing.

The likely cause is that `mlx_lm.server` doesn't enforce the `json_schema`
response format, so the model wrote JSON in a shape of its own choosing,
where `llama-server` constrained every Bonsai answer to the schema.
Non-spending at 0.00 fits too: a missing `isSpending` reads as spending.
Two things confirm or rule it out:

1. The format column in `run.py`'s per-sample lines. Anything other than
   `json_schema` means the schema wasn't enforced.
2. The keys of a few answers in `runs/<name>/predictions.jsonl`.

If the schema wasn't enforced, the fair rerun is the same 4-bit 0.8B as a
GGUF (`Q4_K_M`, about 0.5 GiB) on `llama-server`, the runtime that served
the Bonsai models. 3.6 s per sample is fast, but also unverified: an answer
that ignores the schema may simply be shorter.

## 6. After SFT

Not run yet. Record both scores for each round, the 16-bit one and the
quantized export's (see *What counts as the baseline*). Record each round of `iterate.py` here
(`work/<model>/rounds.jsonl`, `runs/iter_r*/grade.json`), and compare it
as a new column of *The hard cases* in the summary. Those rows show whether training teaches the hard cases rather than
only the output format.
