# How the small model does

The spending agent has to run on a phone, so its model has to ship at 1 GB
or less. These tables score the agent, through the same harness and
settings, with:

- **the teacher**, `qwen3.8-27b-fp8-v5`: Qwen3.8-27B in FP8 on vLLM, the
  model whose sessions the student learns from;
- **the untuned student**, `qwen3.5-0.8b-…`: Qwen3.5-0.8B as released;
- **the trained student**: the same 0.8B after distillation on the
  teacher's sessions (`…-sft-rN`, round N) and then reinforcement learning
  from an SFT round (`…-grpo-rN`).

The last part of a run's name is how it was served: `fp16` or `bf16` on
vLLM, or a llama.cpp GGUF (`q8_0`, `q4_k_m`) through `llama-server`, the
kind of file that ships. The `GGUF` column gives that file's size. Runs on
`dev-novel` end in `-dev-novel`.

`dev` is 280 cases over seven locales: till receipts as OCR reads them,
order emails, bank notifications, wallet screens, casual notes and messages
that aren't spending. `dev-novel` is the same mix at merchants and with item
names invented for it, so no model can have seen them: a model that reads
receipts scores about the same on both, one that remembers shops drops on
`dev-novel`. Neither is ever trained on.

`record_core` is the share of cases where the record is right on whether
it is spending, the amount, the currency, the merchant and the category;
`record_all` adds the date, the time, the payment instrument and the tags.
`false approvals` is the share of cases the app would have approved
without review while the record was wrong, the safety number. *Reading a
grade* in [README.md](README.md) defines every column, and *Running a
baseline* shows how to score a model of your own.

<!-- board:begin -->
Suite `dev` (`ba5103a047a0ddd7`, 280 cases)

Agent contract `7fcba05bea99d624` (the current contract).

| run | GGUF | n | record_core (95% CI) | record_all | reward | amount | merchant | category | tags | false approvals | tools/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-fp8-v5 | — | 274 | 0.887 (0.85–0.92) | 0.792 | 0.959 | 0.99 | 0.98 | 0.90 | 0.96 | 0.175 | 2.19 |
| qwen3.5-0.8b-sft-r2-q8_0 | Q8_0, 0.81 GB | 274 | 0.836 (0.79–0.88) | 0.726 | 0.892 | 0.96 | 0.98 | 0.86 | 0.93 | 0.248 | 2.28 |
| qwen3.5-0.8b-sft-r2-bf16 | — | 274 | 0.821 (0.77–0.87) | 0.715 | 0.883 | 0.95 | 0.98 | 0.86 | 0.92 | 0.259 | 2.02 |
| qwen3.5-0.8b-grpo-r2-bf16 | — | 274 | 0.807 (0.76–0.85) | 0.682 | 0.947 | 0.96 | 0.99 | 0.86 | 0.92 | 0.117 | 1.43 |
| qwen3.5-0.8b-grpo-r2-q4_k_m | Q4_K_M, 0.53 GB | 274 | 0.807 (0.76–0.85) | 0.675 | 0.937 | 0.95 | 0.98 | 0.85 | 0.93 | 0.128 | 1.66 |
| qwen3.5-0.8b-sft-r3-bf16 | — | 274 | 0.807 (0.76–0.85) | 0.708 | 0.892 | 0.96 | 0.98 | 0.86 | 0.94 | 0.226 | 2.61 |
| qwen3.5-0.8b-grpo-r2-q8_0 | Q8_0, 0.81 GB | 274 | 0.803 (0.76–0.85) | 0.697 | 0.950 | 0.96 | 0.98 | 0.86 | 0.93 | 0.113 | 1.65 |
| qwen3.5-0.8b-sft-r2-q4_k_m | Q4_K_M, 0.53 GB | 274 | 0.777 (0.73–0.82) | 0.664 | 0.818 | 0.94 | 0.97 | 0.83 | 0.93 | 0.314 | 3.60 |
| qwen3.5-0.8b-grpo-r1-bf16 | — | 274 | 0.759 (0.71–0.81) | 0.653 | 0.853 | 0.95 | 0.97 | 0.81 | 0.92 | 0.314 | 1.61 |
| qwen3.5-0.8b-sft-r1-bf16 | — | 274 | 0.752 (0.70–0.80) | 0.631 | 0.846 | 0.95 | 0.98 | 0.80 | 0.91 | 0.336 | 0.95 |
| qwen3.5-0.8b-q8_0 | Q8_0, 0.81 GB | 274 | 0.226 (0.18–0.28) | 0.029 | 0.203 | 0.72 | 0.60 | 0.37 | 0.15 | 0.799 | 4.93 |
| qwen3.5-0.8b-fp16 | — | 274 | 0.197 (0.15–0.24) | 0.026 | -0.097 | 0.56 | 0.45 | 0.33 | 0.15 | 0.577 | 5.25 |
| qwen3.5-0.8b-q4_k_m | Q4_K_M, 0.53 GB | 274 | 0.117 (0.08–0.16) | 0.018 | 0.221 | 0.71 | 0.49 | 0.28 | 0.12 | 0.748 | 1.33 |

Suite `dev-novel` (`2b88fe090a58ac9a`, 280 cases)

Agent contract `7fcba05bea99d624` (the current contract).

| run | GGUF | n | record_core (95% CI) | record_all | reward | amount | merchant | category | tags | false approvals | tools/case |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b-grpo-r2-q8_0-dev-novel | Q8_0, 0.81 GB | 275 | 0.793 (0.75–0.84) | 0.636 | 0.927 | 0.97 | 0.98 | 0.83 | 0.86 | 0.145 | 1.62 |
| qwen3.5-0.8b-grpo-r2-bf16-dev-novel | — | 275 | 0.771 (0.72–0.82) | 0.622 | 0.923 | 0.97 | 0.98 | 0.80 | 0.86 | 0.149 | 1.46 |
| qwen3.5-0.8b-grpo-r2-q4_k_m-dev-novel | Q4_K_M, 0.53 GB | 275 | 0.760 (0.71–0.81) | 0.607 | 0.924 | 0.96 | 0.97 | 0.80 | 0.87 | 0.127 | 1.60 |
| qwen3.5-0.8b-sft-r3-bf16-dev-novel | — | 275 | 0.760 (0.71–0.81) | 0.629 | 0.877 | 0.97 | 0.98 | 0.79 | 0.87 | 0.225 | 2.86 |
| qwen3.5-0.8b-grpo-r1-bf16-dev-novel | — | 275 | 0.753 (0.70–0.80) | 0.615 | 0.830 | 0.96 | 0.97 | 0.79 | 0.85 | 0.342 | 1.53 |
| qwen3.5-0.8b-sft-r2-q8_0-dev-novel | Q8_0, 0.81 GB | 275 | 0.745 (0.69–0.80) | 0.611 | 0.809 | 0.95 | 0.97 | 0.79 | 0.87 | 0.353 | 2.33 |
| qwen3.5-0.8b-sft-r2-q4_k_m-dev-novel | Q4_K_M, 0.53 GB | 275 | 0.731 (0.68–0.79) | 0.611 | 0.780 | 0.94 | 0.96 | 0.77 | 0.89 | 0.360 | 3.78 |
| qwen3.5-0.8b-sft-r2-bf16-dev-novel | — | 275 | 0.724 (0.67–0.77) | 0.604 | 0.807 | 0.95 | 0.96 | 0.77 | 0.86 | 0.356 | 1.97 |
| qwen3.5-0.8b-sft-r1-bf16-dev-novel | — | 275 | 0.716 (0.66–0.77) | 0.604 | 0.814 | 0.96 | 0.96 | 0.76 | 0.85 | 0.375 | 0.94 |
<!-- board:end -->
