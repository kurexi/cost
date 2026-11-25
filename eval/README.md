# Spending agent evaluation

A frozen suite of cases, a runner that puts the spending agent
(`spending_agent/`) through each one, and a grader. Together they give a
baseline for every model and show whether a distilled student is closing the
gap to its teacher. Standard library only.

```
make_suite.py   generates a suite: cases (content + gold record) and simulated users (ledger, tags, memory)
receipts.py     the receipt generator under it: multilingual content with gold records; locales.py holds its per-language data
suite.py        loads a suite; turns a user into a fresh ledger folder for one case
run.py          runs the agent on a suite against any OpenAI-compatible server; resumable
grade.py        field-level grading of the settled drafts, breakdowns, safety, the RL reward
compare.py      runs side by side on shared cases: deltas with paired-bootstrap intervals, flips
board.py        the leaderboard; --write updates RESULTS.md, --publish BENCHMARK.md and the evidence in ../README.md
BENCHMARK.md    how the small model does: every graded dev and dev-novel run, without how it was served
backends.py     oracle (answers the gold, must grade perfect) and silent (the floor)
probe_formats.py asks a server which answer schemas it accepts, in seconds, before a run
baseline_t4.ipynb the ~1B baseline (Qwen3.5-0.8B, fp16, vLLM) on a Colab T4, set up like the teacher's run
baseline_gguf_t4.ipynb a 0.8B as a GGUF (base or tuned, Q4_K_M to Q8_0) through llama-server on a Colab T4: the shippable-size scores
sft_data.py     SFT pairs from a teacher's sessions on train suites: every request rebuilt as the agent sent it
train_sft.py    LoRA SFT on those pairs, the loss on completions only
merge_lora.py   merges a train_sft.py adapter into a copy of the base checkpoint, in the layout llama.cpp converts
sft_a100.ipynb  one round of distillation on Colab: teacher on train (A100, first round only), pairs, SFT, merge, optional bf16 score
train_grpo.py   GRPO from an SFT adapter: whole sessions of the real loop, sampled in-process and graded with the reward
grpo.ipynb      GRPO on Colab (A100 or L4): train-rl suite, train_grpo.py, pick a kept step on val-rl, merge, bf16 score
suites/         dev (280 cases), dev-novel (280, invented merchants and items) and holdout (140), committed and frozen
novel_names.py  invents merchants and item names per simulated user, from two disjoint halves (train, eval)
runs/           one folder per run: meta, results, transcripts, grade (in the records folder)
```

**Where runs are kept.** `run.py`, `grade.py`, `compare.py`, `board.py` and
`sft_data.py` read and write `runs/<name>/`, and `board.py --write` the
`RESULTS.md` leaderboard, in the records folder: `$SPENDING_EVAL_RECORDS`
when it is set, otherwise this folder (where both are git-ignored). The runs
behind `BENCHMARK.md`, with their training logs and notes, are in the
repository's `eval-records/`; point `SPENDING_EVAL_RECORDS` there to grade,
compare or extend them, or at a folder of your own for your own runs.
`board.py --write` starts a `RESULTS.md` there if there is none.

## The suites

Content and gold labels come from the receipt generator in
`receipts.py` (with the per-language data in `locales.py`) across seven home locales (en-GB, en-US, zh, ja,
fr, es, de):
- till receipts as OCR reads them;
- order emails;
- bank notifications;
- wallet screens;
- casual notes;
- non-spending messages.

On top of those, `make_suite.py` adds what the agent is judged on:

| What | Traits it shows up under |
|---|---|
| Long content: card slips, VAT tables, loyalty and returns footers, legal text; receipts re-rendered with 10–40 item lines | `long_content`, `many_items` |
| Content over the 4,000-character budget, which must be rejected before any model call | `over_limit` |
| A standing tag on one merchant's records ("Work"), and cases at that merchant | `standing_tag` |
| A trip tag for trips the user is already recording, and purchases abroad with no tag yet (proposing one is allowed there, nowhere else) | `existing_trip_tag`, `new_trip` |
| A decoy tag ("Gifts") on two unrelated records | — |
| Memory: empty, helpful (how a mixed merchant is filed, which name is a card, which merchant is work), or helpful with one wrong line that the records contradict | `memory_helps`, `memory_misleads` |
| Provenance on every ledger record (`manual`, `agent`, `agent_reviewed`) | — |
| Casual notes at a merchant filed two ways accept either category, because nothing in them decides | `category_ambiguous` |

The generator's own traits are kept as well: `no_printed_total`,
`cash_change`, `was_price`, `tax_added`, `card_heading`, `issuer_on_receipt`,
`travel`, `date_without_year`, `relative_date`, `needs_proposal`,
`mixed_merchant`, and the rest.

Each case runs on a fresh copy of its user's ledger, without the records
dated after the case's today. Cases are therefore independent: one case's
memory edit never reaches another.

**Suites are frozen.** Every run records the suite's fingerprint, and runs
on different fingerprints are never compared. `tests/` checks that the
committed `dev` and `dev-novel` suites still regenerate byte for byte from its seed.
Changing the generator therefore means a new suite (`--name dev2`) and new
baselines; it never means editing `dev` in place.

**Names the model can't have seen: `dev-novel`.** The generator's merchants
are a fixed list of about a hundred real shops, each with its own items, and
`dev`, `holdout` and an old-style train suite all draw from it. A model can
therefore learn "TESCO at the top: Tesco, Groceries" from the name alone, and
`dev` would reward that. `novel_names.py` gives each simulated user a world in
which a share of the merchants are invented for them. An invented merchant
keeps the real one's items, prices, category and kind, and each item name is
rewritten too (a new brand, another size or pack, the till's capitals or
abbreviations), while the word that says what the product is stays.
- Invented merchants are built from the same stems as invented card and bank
  names, so the name never says which side of the payment it is on.
- Some names carry no word for their kind; 5% carry another kind's word.
- Every invented string falls in one of two halves by a hash of its folded
  form. Training suites draw from `train`, and `dev-novel`
  (`--novel-share 1.0 --novel-split eval --seed 31`) only from `eval`, so no
  name in it can have been trained on.

A model that reads receipts scores about the same on `dev` and `dev-novel`.
One that remembers shops drops on `dev-novel`, and the size of the drop,
against the teacher's, is what was remembered. The trait `invented_merchant`
marks the cases at an invented merchant, so `compare.py` breaks the score down
by it on any suite.

**Use `holdout` sparingly.** Run it only at milestones: a baseline, or the
end of a distillation round. Never run it while tuning, so a gain on `dev`
can be confirmed on cases nobody has looked at.

For training data, generate a separate split with a different seed, many
users (so each invented shop occurs about once), and half of each user's
merchants and items invented (so real brands stay known):

```bash
python3 make_suite.py --name train --per-locale 300 --users-per-locale 30 --seed 1 --novel-share 0.5
```

That is 2,100 cases for 210 users, fingerprint `a5437ef135b8f48f`.
`suites/train*` is git-ignored.

## Running a baseline

Start the model server first, for example llama-server with Bonsai 27B on
port 8090. Then:

```bash
cd eval
python3 run.py --suite dev --name bonsai-27b --base-url http://127.0.0.1:8090/v1 --model bonsai --reasoning-effort none
```

**On a slow GPU, run several cases at once.** Decoding is limited by memory
bandwidth, so a server with parallel slots batches four sessions for little
more than the cost of one. Start llama-server with `-np 4` and a context of
four slots' worth (`-c 49152`, about 12K tokens each), then pass
`--workers 4`. Batched generation can differ slightly from serial
generation, so `meta.json` records `workers`; keep it the same for runs you
compare. Each case line shows its prefill and decode seconds, and the grade
shows the averages, so you can see where the time goes.

`--limit 70` runs a quick, locale-balanced prefix (the suite interleaves
locales). An interrupted run resumes from where it stopped when you rerun
the same command. It refuses to resume if the suite, the agent's contract
(its instructions, schema and tools) or the settings changed, because the
results would no longer be one run. Each run also records a hash of the
agent's code (`agent_code`): `compare.py` notes when two runs differ in
it, because a tool that answers differently changes what the model sees
even when the contract doesn't.

**Check the answer format first.** The agent asks for the bounded schema
(`json_schema`). If the server refuses it, the agent retries with the same
fields minus the patterns and length caps (`json_schema_plain`). After that
it falls back to `json_object` and then to no format, where the prompt lists
the fields because no grammar names them. The grade prints the formats used
and the server's first refusal. Answers off the schema grammar are much
weaker, so a run on them is not a fair baseline.
`python3 probe_formats.py --base-url … --model …` shows in seconds which
bound a server refuses.

Check the harness before trusting a number. Both of these take about a
second:
- `--backend oracle` must grade 1.000 everywhere;
- `--backend silent` must grade 0.

## Reading a grade

What is graded is the draft the session settled, which is the row the user
would review. The model's raw text is not graded.

- `record_core`: whether it is spending, the amount (exact, in minor
  units), the currency, the merchant and the category.
- `record_all`: everything in `record_core`, plus the date, the time, the
  payment instrument, the tags, and a tag proposal only where one is
  allowed.
- `reward`: the field-weighted scalar the RL step trains on. A missing draft
  scores -1, each wasted tool call costs 0.03 (refused, repeated, unknown or
  invalid), and a fully right record earns a 0.05 bonus. A draft the app
  would approve on its own earns 0.05 more when it is fully right and loses
  0.45 when it isn't, so sounding confident pays only where the draft is
  right at least 9 times in 10 (`APPROVAL_PRECISION`). Both terms are
  needed: with the penalty alone, never approving would be the safest
  policy.
- `false_approval`: the share of cases the app would have approved without
  review (a spending draft with an existing category and confidence ≥ 0.8)
  while the record was wrong. This is the safety number. The audits exist to
  drive it to zero, so watch it as closely as accuracy.
- Process figures come with every grade: failures, tool calls by tool,
  wasted calls, how often the budget was spent, compactions, memory edits,
  seconds and tokens per case.

**What counts as right where the content doesn't decide.** These are
grading rules, so the frozen suite never changes for them, and every run is
regraded under the current rules whenever it is graded:
- **Date.** The grade compares the date the record gets. When nothing gives
  a date, an empty date and the arrival date (the case's today, which the
  app falls back to) are both right.
- **Category.** Goods with no category of their own (clothing, home, pets,
  kids) may go under the user's broad Shopping category, when the user has
  one, instead of a proposal.
- **Trip tags.** A trip tag on a purchase after the trip's last tagged
  record may be left off. Inside the trip's dates it is required.

`record_core`, `record_all` and `reward` come with 95% bootstrap intervals.
On 274 cases, an accuracy near 0.8 is only known to about ±0.05, so read
small differences with that in mind.

## Comparing a student with its baseline

```bash
python3 compare.py bonsai-27b qwen-0.8b-sft-r1 qwen-0.8b-grpo-r2
```

The first run is the baseline. Each later run gets a change per metric, with
a 95% paired-bootstrap interval; `*` marks the intervals that exclude zero.
Then come one metric by language, style and trait, and the cases fixed and
broken between the first and last run, with a sign test. Only cases every
run finished are compared.

For distillation, the teacher's `runs/<name>/transcripts.jsonl` holds every
session in full, which is the material to train on. Keep the sessions
whose case graded `record_all` (in `grade.json` → `per_case`), and train
only on cases from a train suite, never from `dev` or `holdout`.

## Distilling the teacher

The student is Qwen3.5-0.8B and it has to ship at 1 GB or less. The order
is SFT on the teacher's sessions, then GRPO, then quantization, and each
step is scored as a GGUF through `llama-server` on the T4, like the
baselines. `sft_a100.ipynb` runs one SFT round from start to finish. The
teacher's steps run once, on an A100; a later round finds their output on
Drive, skips them, and runs on an L4:

1. the teacher (`Qwen/Qwen3.8-27B-FP8` on vLLM, its `dev` settings) on one
   or more train suites (see *The suites*), for example `train` as
   `runs/teacher-train` and `train-2` (`make_suite.py --name train-2
   --per-locale 600 --users-per-locale 60 --seed 4 --novel-share 0.5`:
   4,200 cases, 420 users, fingerprint `9c208ad58ed4d37a`, git-ignored) as
   `runs/teacher-train-2`, and on `dev-novel` for its row there;
2. `sft_data.py teacher-train teacher-train-2` → one pairs file from every
   run named (with `--doubt-confidence` for the wrong drafts);
3. `train_sft.py` → a LoRA adapter (`LOSS` in the
   notebook: per token);
4. `merge_lora.py` → a merged Hugging Face folder, which
   `baseline_gguf_t4.ipynb` (`SOURCE`, `NAME`) converts and scores on `dev`
   and `dev-novel` as `<NAME>-q8_0`, `<NAME>-q8_0-dev-novel`,
   `<NAME>-q4_k_m`, `<NAME>-q4_k_m-dev-novel`. The untuned 0.8B needs its
   `dev-novel` rows too: the same notebook with `SUITES = ["dev-novel"]`.

**What a pair is.** Every model call in a kept session gives one
(prompt, completion) pair, which is the format `train_sft.py` reads. The
transcript keeps the final conversation and the raw replies, not each
request. `sft_data.py` therefore replays the session: compaction, the
assistant turns and the tool answers, in order. A replay that doesn't end on
the transcript's final conversation is dropped and counted. Tool rounds are
rendered with the tool definitions; the answer is rendered without them,
with `CLOSING`, and with the field guide where the format ladder fell back.
Thinking is off, so each prompt ends in the template's empty think block.
There are three kinds of completion from a kept session, and a fourth from
a wrong one:
- `tool_call`: the calls in the template's canonical form, plus any words
  written before them, since the loop keeps those;
- `end_tools`: an empty reply. The loop drops that text, so the teacher's
  paragraph there would teach decode time spent on nothing;
  `--end-tools-text teacher` keeps it;
- `answer`: the JSON;
- `doubt`: with `--doubt-confidence`, the answer of a session that drafted a
  wrong record, its confidence set to that value. `train_spans` names the
  number's characters, and `train_sft.py` trains the tokens overlapping them
  and nothing else.

The renderer is checked against `transformers` before anything is written,
and pairs over `--max-len` are dropped. The trainer would otherwise cut the
instructions off the start of the prompt. On the teacher's `dev` sessions a
pair is about 2,300 tokens on average and 4,200 at most, so 8,192 leaves
room.

**Why merge into a copy.** `train_sft.py` loads the text model alone
(`Qwen3_5ForCausalLM`, `model.layers.…`). llama.cpp converts the full
checkpoint (`Qwen3_5ForConditionalGeneration`,
`model.language_model.layers.…`, plus the vision tower and the MTP head).
`merge_lora.py` adds each adapter's `B·A·alpha/r` to the one checkpoint
weight it adapts and copies everything else unchanged. Its test checks the
result against `peft`'s own merge on a tiny Qwen3.5.

Of a teacher run, keep `meta.json`, `results.jsonl` and `grade.json` with
your records. Its transcripts, the training material, are tens of
megabytes; keep them on Drive, out of git, and the pairs too. The notebooks
put each round's pair counts, adapter config, training arguments
(`train_args.json`), train log and merge report in `sft/<round>/`.

**Why the loss is averaged per token.** Averaged per sequence, a stop at
the end of the tool phase is a single `<|im_end|>`, so it puts its
sequence's whole weight on the token that decides to stop. A further tool
call spreads its weight over 25 or more tokens, so only a small share lands
on the token that decides to go on, and the student learns to stop after its
first tool round. Averaged per token, the two decisions weigh the same.

**Why the wrong drafts come back as doubt pairs.** Trained only on the
teacher's right sessions, every answer the student is taught is a sure one,
so its confidence ends up as high on wrong drafts as on right ones, and the
drafts it would auto-approve are often wrong. The teacher's own confidence
barely separates right from wrong, so there is little to copy, and no
thinking mode the student could afford on a phone. The gold says which
drafts were wrong, so the wrong ones come back as `doubt` pairs: the
teacher's own answer as context, with a low confidence as the only target.
The wrong fields are never trained on. Under the per-token loss the
confidence number weighs the same in a right answer and a doubt pair, so
the student is taught the confidence in proportion to how often drafts like
this one were right.

## GRPO

`grpo.ipynb` continues an SFT adapter with `train_grpo.py` on `train-rl`
(`make_suite.py --name train-rl --per-locale 100 --users-per-locale 10
--seed 2 --novel-share 0.5`, 700 cases, fingerprint `3b7e550726f6949b`,
git-ignored). Each step runs 8 cases × 8 sampled sessions and grades each
session with `grade.reward`. The student is moved towards the sessions that
beat their group's mean, with one update per batch of sessions.

**The sessions are the real loop.** `PolicyBackend` is `ChatBackend` with
the network replaced by the student: each request is rendered with the chat
template the way a server renders it and generated in this process. So
compaction, the tools, the budget and the answer ladder are the ones `run.py`
measures. Sessions run in threads, and a `Batcher` generates the requests of
every session waiting at the same time in one batch.

**The answer is sampled under its grammar.** vLLM compiles a request's
JSON schema with xgrammar, and `AnswerGrammar` does the same while sampling,
masking every answer row's logits; tool rounds are free text, as served.
Two differences, both deliberate: a run of whitespace between elements is
capped at 32 characters (at temperature 1 an answer can fall into
whitespace for good, and one such row holds a static batch for the whole
answer budget), and the schema is compiled as written, never re-serialised
with sorted keys, which would reorder its properties and put `reasoning`
out of first place. Sampled without a grammar (`--no-grammar`), with an
answer outside the schema graded as no draft, most of the reward's rise is
the student learning the format a served model gets for free. An answer cut
off at the token limit is still no draft.

**The confidence is credited with the approval alone.** The reward's
approval term (`grade.approval_term`: +0.05 for an approved right draft,
-0.45 for an approved wrong one, nothing for one left to review) gets its
own group-normalised advantage, applied to the tokens that write the
answer's confidence number (`confidence_positions`, only when it ends a
finished answer), averaged per session and weighed like all of the
session's other tokens together (`--confidence-weight`). The rest of the
reward goes to every other completion token. With one advantage over the
whole session, the confidence stays as high on wrong drafts as on right
ones.

**The rest of the loss is averaged per token,** for the reason SFT's
is: averaged per completion, a one-token stop would carry its whole weight
on the token that decides it. A session's tool rounds extend one another
token for token, so they are scored in one forward pass. The answer adds a
closing turn that re-renders the earlier ones, so it is scored on its own.

**A KL term holds the student near where it started** (`--kl`, 0.04 by
default, the k3 estimator per token). The starting adapter is loaded beside
the trained one as a frozen `reference` adapter and scores the same tokens.
Without it, categories drift on real shops. LoRA dropout is
off, so the tokens are scored by the same policy that sampled them.

**The last step isn't assumed to be the best.** False approvals can be
lowest early in training and creep back up as the confidence rises, so
`--keep-every 20` keeps the adapter of every 20th step and of the last as
`step-NNN/` (a whole copy of the checkpoint's adapter, renamed into place;
a resume that lands on a kept step makes a missing copy from the
checkpoint). The notebook then scores each kept step on `val-rl`
(`make_suite.py --name val-rl --per-locale 40 --users-per-locale 10
--seed 3 --novel-share 0.5`, fingerprint `99159aeb92055fbe`, git-ignored;
the 242 of its 280 cases whose content no frozen suite and not `train-rl`
repeats). It takes the step with the highest reward and records it in
`chosen.json`. Choosing on `dev` and then reporting `dev` would report the
luckiest of the candidates, so `dev` and `dev-novel` only ever see the
chosen one. `val-rl` can't be trained on: the train scripts take only
suites named `train…`.

**No frozen content.** The generators draw non-spending messages from a
small pool, so 127 of `train`'s cases repeat a `dev` case word for word,
and 53 of `train-rl`'s repeat a case of one of the frozen suites. All of
them are non-spending. `train_grpo.py` and `sft_data.py` leave those out
(`frozen_content` in `sft_data.py`'s stats: 173 of `train`, 344 of
`train-2`, against every frozen suite).

## Tests

```bash
cd eval && python3 -m unittest discover -s tests -t .
```
