# Spending agent results

Every graded run on the `dev` and `dev-novel` suites, one table each, best `record_core` first. Regenerate
the tables with `python3 board.py --write` after a run; numbers on a
different suite fingerprint are not comparable and are listed apart.

Run the `holdout` suite only at milestones (a baseline, a finished
distillation round), never while tuning prompts or training settings, so it
stays a fair check that a gain on `dev` is real.

<!-- board:begin -->
Suite `dev` (`ba5103a047a0ddd7`, 280 cases)

| run | model | contract | n | record_core (95% CI) | record_all | reward | amount | merchant | category | tags | false approvals | s/case | tools/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-fp8-v5 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.887 (0.85–0.92) | 0.792 | 0.959 | 0.99 | 0.98 | 0.90 | 0.96 | 0.175 | 39.0 | 2.19 |
| qwen3.5-0.8b-sft-r2-q8_0 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.836 (0.79–0.88) | 0.726 | 0.892 | 0.96 | 0.98 | 0.86 | 0.93 | 0.248 | 16.5 | 2.28 |
| qwen3.5-0.8b-sft-r2-bf16 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.821 (0.77–0.87) | 0.715 | 0.883 | 0.95 | 0.98 | 0.86 | 0.92 | 0.259 | 7.9 | 2.02 |
| qwen3.5-0.8b-grpo-r2-bf16 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.807 (0.76–0.85) | 0.682 | 0.947 | 0.96 | 0.99 | 0.86 | 0.92 | 0.117 | 7.6 | 1.43 |
| qwen3.5-0.8b-grpo-r2-q4_k_m | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.807 (0.76–0.85) | 0.675 | 0.937 | 0.95 | 0.98 | 0.85 | 0.93 | 0.128 | 22.1 | 1.66 |
| qwen3.5-0.8b-sft-r3-bf16 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.807 (0.76–0.85) | 0.708 | 0.892 | 0.96 | 0.98 | 0.86 | 0.94 | 0.226 | 9.3 | 2.61 |
| qwen3.5-0.8b-grpo-r2-q8_0 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.803 (0.76–0.85) | 0.697 | 0.950 | 0.96 | 0.98 | 0.86 | 0.93 | 0.113 | 22.1 | 1.65 |
| qwen3.5-0.8b-sft-r2-q4_k_m | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.777 (0.73–0.82) | 0.664 | 0.818 | 0.94 | 0.97 | 0.83 | 0.93 | 0.314 | 26.8 | 3.60 |
| qwen3.5-0.8b-grpo-r1-bf16 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.759 (0.71–0.81) | 0.653 | 0.853 | 0.95 | 0.97 | 0.81 | 0.92 | 0.314 | 7.1 | 1.61 |
| qwen3.5-0.8b-sft-r1-bf16 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.752 (0.70–0.80) | 0.631 | 0.846 | 0.95 | 0.98 | 0.80 | 0.91 | 0.336 | 6.7 | 0.95 |
| qwen3.5-0.8b-q8_0 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.226 (0.18–0.28) | 0.029 | 0.203 | 0.72 | 0.60 | 0.37 | 0.15 | 0.799 | 70.4 | 4.93 |
| qwen3.5-0.8b-fp16 | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.197 (0.15–0.24) | 0.026 | -0.097 | 0.56 | 0.45 | 0.33 | 0.15 | 0.577 | 47.6 | 5.25 |
| qwen3.5-0.8b-q4_k_m | qwen (thinking: none) | `7fcba05bea99d624` | 274 | 0.117 (0.08–0.16) | 0.018 | 0.221 | 0.71 | 0.49 | 0.28 | 0.12 | 0.748 | 62.0 | 1.33 |

Suite `dev-novel` (`2b88fe090a58ac9a`, 280 cases)

| run | model | contract | n | record_core (95% CI) | record_all | reward | amount | merchant | category | tags | false approvals | s/case | tools/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b-grpo-r2-q8_0-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.793 (0.75–0.84) | 0.636 | 0.927 | 0.97 | 0.98 | 0.83 | 0.86 | 0.145 | 22.5 | 1.62 |
| qwen3.5-0.8b-grpo-r2-bf16-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.771 (0.72–0.82) | 0.622 | 0.923 | 0.97 | 0.98 | 0.80 | 0.86 | 0.149 | 7.5 | 1.46 |
| qwen3.5-0.8b-grpo-r2-q4_k_m-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.760 (0.71–0.81) | 0.607 | 0.924 | 0.96 | 0.97 | 0.80 | 0.87 | 0.127 | 22.6 | 1.60 |
| qwen3.5-0.8b-sft-r3-bf16-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.760 (0.71–0.81) | 0.629 | 0.877 | 0.97 | 0.98 | 0.79 | 0.87 | 0.225 | 9.6 | 2.86 |
| qwen3.5-0.8b-grpo-r1-bf16-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.753 (0.70–0.80) | 0.615 | 0.830 | 0.96 | 0.97 | 0.79 | 0.85 | 0.342 | 6.9 | 1.53 |
| qwen3.5-0.8b-sft-r2-q8_0-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.745 (0.69–0.80) | 0.611 | 0.809 | 0.95 | 0.97 | 0.79 | 0.87 | 0.353 | 16.8 | 2.33 |
| qwen3.5-0.8b-sft-r2-q4_k_m-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.731 (0.68–0.79) | 0.611 | 0.780 | 0.94 | 0.96 | 0.77 | 0.89 | 0.360 | 26.8 | 3.78 |
| qwen3.5-0.8b-sft-r2-bf16-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.724 (0.67–0.77) | 0.604 | 0.807 | 0.95 | 0.96 | 0.77 | 0.86 | 0.356 | 7.9 | 1.97 |
| qwen3.5-0.8b-sft-r1-bf16-dev-novel | qwen (thinking: none) | `7fcba05bea99d624` | 275 | 0.716 (0.66–0.77) | 0.604 | 0.814 | 0.96 | 0.96 | 0.76 | 0.85 | 0.375 | 6.5 | 0.94 |
<!-- board:end -->

## Notes per run

Add a line per run: what was run, on what hardware, anything unusual.

- `qwen3.8-27b-fp8-v5`: the teacher baseline. Qwen3.8 27B in FP8 on vLLM,
  Colab A100 40GB, 16 cases in flight (`--max-num-seqs 32`), thinking off,
  all 280 cases in 11.6 minutes. Every answer used the bounded
  `json_schema`. One case failed: the model closed its reasoning string
  with a quote it meant as punctuation, then padded with whitespace.
  Earlier attempts on the same model are not listed. v1–v3 were broken
  by schema problems (truncation, a bound vLLM refused, and fallback
  formats that didn't name the fields); v4's amount pattern blanked 21% of
  amounts. Under the first grading rules it scored 0.821 `record_core`
  and 0.653 `record_all`. Then three judgement calls were settled (see
  the README): the arrival date when none is given, Shopping for goods
  with no category of their own, and trip tags after the trip's last
  record. They moved it to 0.887 and 0.792.
- `qwen3.5-0.8b-fp16`: the ~1B baseline the student is measured against.
  Qwen3.8 has nothing below 27B, so this is untuned Qwen3.5-0.8B, the
  nearest Qwen and the student's starting point. fp16 on vLLM (a T4 has
  neither FP8 nor bf16), Colab T4, `--enable-auto-tool-choice
  --tool-call-parser qwen3_coder --reasoning-parser qwen3`, 16 cases in
  flight (`--max-num-seqs 32`), thinking off, the teacher's limits, all 280
  cases in 13.8 minutes (`baseline_t4.ipynb`). Every answer used the
  bounded `json_schema`. 75 cases (27%) failed the same way: a string
  reached its length cap mid-sentence (`reasoning` at 700 characters in
  62, `note` at 160 in 12, `paymentInstrument` in 1), the grammar closed
  it, and the model padded with whitespace to the token limit. It also
  wastes 2.4 tool calls a case and spends the budget in 43% of cases.
  False approvals are 0.577: on 58% of cases the app would have approved a
  wrong record, because the model reports high confidence when it is wrong.
- `qwen3.5-0.8b-q4_k_m`: the shippable-size baseline. The same untuned
  Qwen3.5-0.8B, converted here to a `Q4_K_M` GGUF (0.498 GiB, no MTP head;
  file and hash in `server.json`) and served by llama.cpp's prebuilt CUDA
  `llama-server` (release `b12400`), Colab T4, 16 slots of 16,384 tokens,
  thinking off, the teacher's limits, all 280 cases in 18.4 minutes
  (`baseline_gguf_t4.ipynb` with `QUANT = "Q4_K_M"`). Every answer used
  the bounded `json_schema` and none failed: llama.cpp's schema grammar
  bounds the whitespace between fields, so the length-cap runaway that
  failed 27% of the fp16 run can't happen here. That makes its `reward` look better than fp16's
  and its `record_core` look closer than it is: on the 199 cases both
  runs answered it scores 0.121 against fp16's 0.271. The loss is in
  looking things up. Both servers gave the model the same prompt
  (first-round prompt tokens are identical on every case), but this run
  called a tool in its first round in 67 cases against fp16's 211, made 32
  record queries against 716, and spent its calls appending the content
  itself to memory (226 rejected as invalid edits). Merchant (0.47
  against 0.62 on those 199) and category (0.29 against 0.45) fall the
  most; amount holds (0.72 against 0.77). False approvals rose to 0.748.
  The runtime changed along with the precision, so this doesn't yet say
  how much is the 4-bit weights; a `Q8_0` run on the same server would.
  Prefix reuse made prefill negligible (0.8 s a case); decode was 60.7 s
  a case, about 22 tokens a second per session at 16 in flight.
- `qwen3.5-0.8b-q8_0`: the control for the 4-bit run. Same model,
  converter, server, release and settings (`baseline_gguf_t4.ipynb` with
  `QUANT = "Q8_0"`), only the quantization changed: `Q8_0`, 0.81 GB
  (0.756 GiB), all 280 cases in 20.8 minutes. It behaves like fp16: a
  tool in the first round in 200 cases against fp16's 211, 720 record
  queries against 716, and on the 199 cases fp16 answered 0.236
  `record_core` against fp16's 0.271 (merchant 0.59 against 0.62,
  category 0.38 against 0.45). Over all 274 it scores 0.226, level with
  fp16 within the interval (+0.029, CI −0.018 to +0.077), helped by
  llama.cpp's grammar never failing an answer. So the server isn't what
  costs the 4-bit run its tool use; the 4-bit weights are. Generation
  speed is the same as the 4-bit run's (211 against 206 tokens a second
  overall), so the slower runs than vLLM's are the server's batching,
  not the weights. False approvals are 0.799, the worst of the three.
- `teacher-train` (not on the board: `train` is training material, not a
  benchmark): the teacher's sessions the first student learns from. The
  `qwen3.8-27b-fp8-v5` settings on the generated `train` suite (2,100
  cases, 210 users, half the merchants and items invented,
  `a5437ef135b8f48f`), Colab A100 80GB, all cases in 77 minutes.
  `record_core` 0.832, `record_all` 0.739, false approvals 0.200: close to
  its `dev` row, so the generated suite is about as hard. `sft_data.py`
  kept the 1,523 `record_all` sessions, replayed every one exactly and
  rendered 5,833 pairs (2,820 tool calls, 1,490 ends of the tool phase,
  1,523 answers; 13.2M tokens, none over 8,192), in `sft/qwen3.5-0.8b-sft-r1/`.
- `qwen3.5-0.8b-sft-r1-bf16`: the first student, unquantized. Qwen3.5-0.8B
  with a rank-16 LoRA on every linear layer (`train_sft.py`, 1 epoch, 730
  steps of 8 pairs, lr 2e-4, 53 minutes at 4,150 tokens a second on the
  A100 80GB), merged into the base checkpoint (186 modules) and served at
  bf16 on vLLM, Colab L4, the teacher's settings, all 280 cases in 1.9
  minutes. `record_core` goes from the base model's 0.197 to 0.752, and
  `record_all` from 0.026 to 0.631. It never failed an answer and never
  wasted a tool call. Two things keep it below the teacher. **It stops
  after one tool round on every case.** 260 of 280 sessions are one
  round of `queryMerchants` and then the end of the tool phase; it never
  calls `queryCategories`, `memory` or the calculator. The teacher goes
  on to a second round of calls in 147 cases. On those cases the student
  scores 0.714 (category 0.78) against 0.795 (0.82) on the rest. The
  likely cause is the loss weighting: it is averaged per pair, so the
  end of the tool phase, a single `<|im_end|>` token, puts its whole
  weight on the decision to stop, while a tool call spreads its weight
  over 25 or more tokens and only a small share of it lands on the
  decision to go on. **Its confidence doesn't follow its accuracy.** It
  reports 0.95 on its right answers and 0.93 on its wrong ones (the
  teacher: 0.94 and 0.88), and 95 of its 99 wrong spending drafts are at
  0.8 or above, so false approvals double to 0.336. It learned only from
  sessions the teacher got right, so it never saw a low confidence.
- `qwen3.5-0.8b-sft-r1-bf16-dev-novel`: the same model and server on
  `dev-novel`, 280 cases in 1.9 minutes. 0.716 `record_core` against
  0.752 on `dev`, inside both intervals: on merchants and items it has
  never seen it reads about as well as on real ones (merchant 0.96 against
  0.98). The teacher's `dev-novel` row, which this gap is meant to be set
  against, is missing: that run wasn't in the results zip.
- `qwen3.5-0.8b-sft-r2-bf16`: round 2, the same pairs and settings as round
  1 with one change: the loss is averaged per token (`--loss token`)
  instead of per sequence. 730 steps in 68 minutes at 3,220 tokens a second
  on a Colab L4. It was scored on the same L4 and server, all 280 cases in
  2.3 minutes. The fix worked on what it was for: 2.87 model rounds a case
  against round 1's 1.95 and the teacher's 2.90, and it now calls
  `queryCategories` (0.55 a case) and `queryRecords` (0.58). `record_core`
  rose from 0.752 to 0.821 (+0.069, CI +0.026 to +0.113), category from
  0.80 to 0.86, `record_all` from 0.631 to 0.715, and false approvals fell
  from 0.336 to 0.259. The biggest gains were French (0.57 to 0.81) and
  cash receipts with change (0.70 to 0.87). It overshoots in one place: in
  34 cases (12%) it walks the category list one title at a time until the
  tool budget ends the phase (the teacher spends its budget in 1%). Those
  cases score 0.71 against 0.84 for the rest, and they are most of its
  0.38 wasted calls a case. It still never calls `memory` or the
  calculator.
- `qwen3.5-0.8b-sft-r2-bf16-dev-novel`: the same model and server on
  `dev-novel`, 280 cases in 2.3 minutes. 0.724 `record_core` against round
  1's 0.716 (+0.007, CI −0.040 to +0.055): no measurable gain, though tool
  use changed the same way (2.81 rounds a case, 39 cases spending the
  budget). The gap to its own `dev` score widened from 0.036 to 0.097.
  Whether that gap is the student remembering names or `dev-novel` being
  harder needs the teacher's row on this suite, which is still missing.
- `qwen3.5-0.8b-sft-r2-q8_0`: the round-2 student at shippable size. The
  merged bf16 model converted to a `Q8_0` GGUF (0.81 GB, 0.756 GiB, under
  the 1 GB limit; hash in `server.json`) and served by `llama-server`
  `b12400` with the baselines' flags (`baseline_gguf_t4.ipynb`, run on a
  Colab L4, not a T4), all 280 cases in 4.8 minutes. 8-bit costs nothing
  measurable: 0.836 `record_core` against 0.821 at bf16 (+0.015, CI −0.011
  to +0.040), false approvals 0.248 against 0.259. It runs a few more tool
  rounds than under vLLM (3.08 a case against 2.87) and spends the budget
  more often (18% against 12%), so the category-walking loop is if anything
  a little worse on llama.cpp. Against the untuned model at the same
  quantization and server, `record_core` goes from 0.226 to 0.836.
- `qwen3.5-0.8b-sft-r2-q8_0-dev-novel`: the same file and server on
  `dev-novel`, 280 cases in 4.9 minutes. 0.745 against 0.724 at bf16
  (+0.022, CI 0.000 to +0.047), false approvals 0.353 against 0.356.
- `qwen3.5-0.8b-sft-r2-q4_k_m`: the round-2 student as a `Q4_K_M` GGUF
  (0.53 GB, 0.498 GiB; hash in `server.json`) from the same bf16 GGUF and
  server as the `Q8_0` row, run on a Colab T4, all 280 cases in 7.8
  minutes. 4 bits cost something real here: 0.777 `record_core` against
  0.836 at `Q8_0` (−0.058, CI −0.099 to −0.022) and 0.821 at bf16 (−0.044,
  CI −0.080 to −0.007), false approvals up to 0.314 from 0.248. The loss is
  in the session more than the reading: the model calls tools more (3.60 a
  case against 2.28), runs 4.0 rounds against 3.0 and spends the budget in
  40% of cases against 17%, walking `queryCategories` and `queryRecords`,
  and category accuracy falls (0.83 against 0.87). German and French and
  wallet screens drop most. The speed (26.8 s a case) is a T4's and not
  comparable with the L4 rows. Against the untuned model at `Q4_K_M`,
  +0.661.
- `qwen3.5-0.8b-sft-r2-q4_k_m-dev-novel`: the same file on `dev-novel`,
  8.0 minutes. 0.731 against 0.745 at `Q8_0` (−0.015, CI −0.051 to
  +0.022) and 0.724 at bf16, false approvals 0.360. Level with the other
  two here, with the same doubled budget spending (39% of cases), so the
  `dev` drop may be partly that suite's draw, but the looping is on both.
- `qwen3.5-0.8b-grpo-r1-bf16`: GRPO round 1 from the round-2 SFT adapter
  (`grpo.ipynb` on a Colab L4: `train-rl`, 80 steps of 8 cases × 8
  sessions, temperature 1.0, lr 1e-5, about 9 hours with `causal-conv1d`
  from step 11), merged and served by vLLM like the SFT rows. Not an
  improvement: 0.759 `record_core` against 0.821 (−0.062, CI −0.102 to
  −0.022), false approvals up to 0.314 from 0.259, approval precision 0.63
  against 0.69. 21 of the 26 cases it broke are the category, a third of
  them German. It calls fewer tools (1.61 a case against 2.02, mostly fewer
  `queryCategories` and `queryRecords`), but the broken cases made almost
  the same calls as before, so that isn't the cause. The training log says
  why the round taught little: the rise in training reward (0.37 to about
  0.70) is almost all answers coming back inside the schema (16% outside
  at the start, 2% at the end), which a served model's grammar enforces
  anyway, while `record_core` on the rollouts stayed near 0.66 and false
  approvals near 0.38. Confidence is not what moved: about 0.94 on right
  and wrong drafts alike, as for the SFT student (0.936 against 0.934), and
  the teacher itself barely separates them (0.931 against 0.895, approval
  precision 0.79), so there was little to copy and the one number per
  session that decides approval is lost among ~300 tokens sharing one
  advantage.
- `qwen3.5-0.8b-grpo-r1-bf16-dev-novel`: the same model on `dev-novel`.
  0.753 against 0.724 (+0.029, CI 0.000 to +0.058), 11 of the 14 cases it
  fixed are the category; false approvals 0.342 against 0.356. A small,
  borderline gain on invented shops against a clear loss on real ones.
- `qwen3.5-0.8b-sft-r3-bf16`: SFT round 3, round 2's data and settings
  plus 529 `doubt` pairs (`sft_data.py --doubt-confidence 0.5`): the
  teacher's wrong drafts on `train`, confidence set to 0.5 and only that
  number trained. 796 steps (the log in `sft/qwen3.5-0.8b-sft-r3/` starts
  at the resume from step 300). `record_core` level with round 2 (0.807
  against 0.821, CI −0.058 to +0.029); reward 0.892 against 0.883, false
  approvals 0.226 against 0.259, approval precision 0.70 against 0.69. The
  confidence learned two values, 0.5 (38 cases) or 0.95 and above, but on
  `dev` they barely sort right from wrong: 0.898 on right drafts against
  0.871 on wrong ones, and 24 of the 38 at 0.5 were right (AUC of the
  confidence, right against wrong, 0.56, round 2 0.55, the teacher 0.66).
  Unplanned: `queryRecords` calls doubled (1.19 a case against 0.57) and
  the budget is spent in 20% of cases against 12%, though the tool-round
  pairs are round 2's.
- `qwen3.5-0.8b-sft-r3-bf16-dev-novel`: the same model on `dev-novel`,
  where the doubt pairs work: 0.760 against 0.724 (+0.036, CI −0.007 to
  +0.076), reward 0.877 against 0.807 (+0.070, outside the interval),
  false approvals 0.225 against 0.356, approval precision 0.66 against
  0.57; confidence 0.897 on right drafts and 0.791 on wrong ones (AUC
  0.65). Half of `train`'s merchants are invented, so its wrong drafts are
  often at shops the student can't know, and that is the doubt it learned.
- `qwen3.5-0.8b-grpo-r2-bf16`: GRPO round 2 from SFT round 3, 80 steps
  (78 logged; two were lost to a resume): answers sampled under the
  extraction grammar, the approval term credited to the confidence tokens
  alone, k3 KL 0.04 to the SFT adapter, LoRA dropout 0 (settings in
  `grpo/qwen3.5-0.8b-grpo-r2/`). `record_core` level with SFT round 3
  (0.807 against 0.807, CI −0.036 to +0.033); reward 0.947 against 0.892
  (+0.055, outside the interval), false approvals 0.117 against 0.226,
  approval precision 0.75 against 0.70. Most of the reward is caution:
  it approves 46% of cases against 75% (the teacher 83%), so right
  approvals fall too, 0.347 against 0.522. Confidence now separates
  better (0.778 on right drafts, 0.675 on wrong ones, AUC 0.63 against
  0.56, the teacher 0.66), taking four values: 0.5 (134 cases), 0.95,
  0.98 and 1.0. The tool looping of round 3 is gone: 1.43 calls a case
  against 2.61, the budget spent in 4% of cases against 20%, 7.6 s a case.
  Over training false approvals were lowest at steps 11–20 (0.061) and
  rose to 0.233 by steps 71–80 as confidence crept back up; the KL stayed
  under 0.005 and no sampled answer left the grammar.
- `qwen3.5-0.8b-grpo-r2-bf16-dev-novel`: the same model on `dev-novel`.
  0.771 against SFT round 3's 0.760 (+0.011); reward 0.923 against 0.877
  (+0.046, outside the interval), false approvals 0.149 against 0.225,
  approval precision 0.69 against 0.66, AUC 0.63; approves 48% against 67%.
- `qwen3.5-0.8b-grpo-r2-q8_0`: GRPO round 2 at shippable size, the merged
  bf16 model converted to a `Q8_0` GGUF (0.81 GB, 0.756 GiB; hash in
  `server.json`) and served by `llama-server` `b12400` with the baselines'
  flags, on a Colab T4 (SFT round 2's `Q8_0` row ran on an L4, so the
  seconds a case, 22.1 against 16.5, are the GPU's, not the model's).
  8-bit costs nothing measurable again: 0.803 `record_core` against 0.807
  at bf16, reward 0.950 against 0.947, false approvals 0.113 against 0.117,
  approval precision 0.77 against 0.75, 48% approved against 46%. A few
  more tool rounds than under vLLM (2.57 a case against 2.36, the budget
  spent in 6% against 4%), the gap SFT round 2 showed too, but far from
  its loop (18% at `Q8_0`). Against the shipped candidate before it,
  `qwen3.5-0.8b-sft-r2-q8_0`: reward 0.950 against 0.892 (+0.058, outside
  the interval), false approvals 0.113 against 0.248, `record_core` 0.803
  against 0.836 (−0.033, inside the interval), and it approves 48% of
  cases against 85%.
- `qwen3.5-0.8b-grpo-r2-q8_0-dev-novel`: the same file and server on
  `dev-novel`. 0.793 against 0.771 at bf16; reward 0.927, false approvals
  0.145, precision 0.70. Against `qwen3.5-0.8b-sft-r2-q8_0-dev-novel`:
  `record_core` +0.047, reward 0.927 against 0.809 (+0.117), false
  approvals 0.145 against 0.353, both outside their intervals.
- `qwen3.5-0.8b-grpo-r2-q4_k_m`: the same merged model as a `Q4_K_M` GGUF
  (0.53 GB), same server and T4. Round 2 at 4-bit loses less than SFT round
  2 did: `record_core` 0.807, level with `Q8_0` (0.803) and bf16 (0.807);
  reward 0.937 against 0.950 at `Q8_0` (−0.012, inside the interval), false
  approvals 0.128 against 0.113. The one field it measurably loses is the
  card or wallet (`instrument` 0.927 against 0.949 at `Q8_0`, outside the
  interval). The tool loop that 4-bit gave SFT round 2 (4.08 rounds a case,
  the budget spent in 41%) doesn't come back: 2.60 rounds, the budget spent
  in 5%, as at `Q8_0`. Against `qwen3.5-0.8b-sft-r2-q4_k_m`: reward 0.937
  against 0.818 (+0.119, outside the interval), `record_core` +0.029.
- `qwen3.5-0.8b-grpo-r2-q4_k_m-dev-novel`: the same file on `dev-novel`.
  0.760 against 0.793 at `Q8_0` (−0.033, CI −0.062 to −0.004) but 0.771
  at bf16 (−0.011), so part of that gap is `Q8_0` drawing high; the
  `instrument` field loses again (0.916 against 0.956) and the merchant a
  little (0.967 against 0.982). Reward 0.924 against 0.927, false
  approvals 0.127 against 0.145, the budget spent in 5%.

## Distillation rounds

What each round of training changed and why, from the results above.

**Why round 2 averages the loss per token.** Round 1 averaged it per
sequence. A stop at the end of the tool phase is a single `<|im_end|>`, so
it puts its sequence's whole weight on the token that decides to stop. A
further tool call spreads its weight over 25 or more tokens, so only a small
share lands on the token that decides to go on. The round-1 student stopped
after its first tool round on 260 of 280 `dev` cases, where the teacher went
on in 147, and it scored 0.714 on those 147 against 0.795 on the rest.
Averaged per token, the two decisions weigh the same.

**Why round 3 adds the wrong drafts.** Rounds 1 and 2 kept only the
teacher's right sessions, so every answer the student was taught was a sure
one. On `dev` the round-2 student's confidence was 0.936 on right drafts and
0.934 on wrong ones, and 31% of the drafts it would auto-approve were
wrong. GRPO didn't fix it (round 1: precision 0.63), and the teacher's own
confidence barely separates right from wrong (0.931 against 0.895), so
there is little to copy and no thinking mode the student could afford on a
phone. The gold says which drafts were wrong, so the wrong ones come back
as `doubt` pairs: the teacher's own answer as context, with a low
confidence as the only target. The wrong fields are never trained on.
Under the per-token loss the confidence number weighs the same in a right
answer and a doubt pair, so the student is taught the confidence in
proportion to how often drafts like this one were right.

**Why round 4 is more data.** GRPO round 2 nearly matched the teacher's
reward on `dev` (0.947 against 0.959) but not its drafts (`record_core`
0.807 against 0.887): the teacher got 40 cases right that the student got
wrong, the reverse 18 times, mostly the category, tags, the amount and the
date on receipts with no printed total, OCR noise, trips and cash change.
Two GRPO rounds kept `record_core` level at best, and rounds 1–3 learned
from 1,523 right sessions. So round 4 adds the teacher on `train-2`, about
three times the sessions, with round 3's recipe otherwise unchanged.

**Frozen content in early SFT rounds.** SFT rounds 1–3 trained on `train` before the repeated
non-spending content was noticed, so their `non_spending` scores may be partly memory. From round 4
`sft_data.py` leaves it out.

**GRPO round 2's log.** False approvals were lowest at steps 11–20 and back up by the end, which is
why `--keep-every` and the `val-rl` choice exist.

**GRPO round 1** sampled without a grammar and graded an answer outside the schema as no draft; most
of its reward's rise was the student learning the format a served model gets for free. One advantage
covered the whole session and the confidence stayed at about 0.94 on right and wrong drafts alike. It
had no KL term, and its categories drifted on real shops.

**The reward's approval terms.** Every committed run was regraded when the approval terms were added
to `reward`, so the `reward` column is one definition throughout.

**What round 2 taught.** Round 2 kept `record_core` (0.807 on `dev`, level with SFT round 3), raised the reward from 0.892 to 0.947, halved false approvals (0.226 to 0.117) and made the confidence separate right drafts from wrong ones (AUC 0.56 to 0.63, the teacher 0.66). But its training log had false approvals lowest at steps 11–20 (0.061 of sessions) and back at 0.233 by steps 71–80, as the confidence crept up again: the last step need not be the best one.

**Why round 4 is more data.** GRPO round 2 nearly matches the teacher's reward (0.947 against 0.959 on `dev`) but not its drafts: `record_core` 0.807 against 0.887. The teacher gets 40 `dev` cases right that the student gets wrong, and the reverse happens 18 times: the category (28 student-only misses against 17), tags (13 against 3), the amount (12 against 2) and the date (8 against 3), mostly on receipts with no printed total, OCR noise, trips, cash change and subtotal lines. Two rounds of GRPO kept `record_core` level at best, and rounds 1–3 learned from 1,523 right sessions. That is skill the teacher has and hasn't handed down, and more of its sessions is the direct way to hand it down. Everything else is round 3's, so the difference between the rounds is the data.
