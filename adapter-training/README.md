# Receipt-extraction adapter training

This directory is the training side of the custom LoRA adapter that
`SpendingAgent` loads for receipt extraction (see `extractionAdapterName` in
`cost 2/Intelligence/SpendingAgent.swift`). The adapter specializes Apple's
on-device foundation model for our exact extraction task — it does not make
the model bigger, but for a narrow task like this it is the largest quality
win available without sending data off-device.

**Privacy:** training uses only the synthetic data generated here. No user
data is involved at any point, training runs on your own machine, and the
trained adapter executes on-device exactly like the base model.

## The version treadmill (read this first)

An `.fmadapter` is compatible with **one specific base model version**. Every
OS update that ships a new base model requires retraining with the matching
toolkit version and publishing a new adapter. The app is built to tolerate
this: when no compatible adapter loads, `SpendingAgent` silently falls back
to the base system model, so a stale adapter is a quality regression, never
a breakage. Budget for retraining before committing to shipping this.

## Prerequisites

- Apple Developer account; download the **Adapter Training Toolkit for
  Apple Intelligence** from developer.apple.com. Pick the toolkit version
  matching the OS/base-model version you target.
- Python + PyTorch on an Apple Silicon Mac with plenty of RAM (or a
  CUDA Linux box) — see the toolkit's own requirements.

## Workflow

1. **Generate data** (stdlib-only script, mirrors the app's exact prompt
   assembly and the `ReceiptExtraction` field names):

   ```sh
   python3 make_training_data.py --count 4000 --out data/train.jsonl --seed 1
   python3 make_training_data.py --count 400  --out data/eval.jsonl  --seed 2
   ```

   Each JSONL line is a `[system, user, assistant]` message list; the
   assistant turn is the gold extraction as JSON. Check the toolkit release
   you downloaded for its expected schema and adapt if it differs.

   Five content styles are sampled: till receipt, order confirmation, bank
   notification, wallet transaction screenshot and casual note. The wallet
   style earns its place — the card is the heading, in the largest type,
   exactly where a till receipt prints the shop, so a model reading position
   rather than meaning answers the card to "who was paid". Its gold answer
   puts the card in `paymentInstrument`, the shop in `merchant`, and says the
   difference out loud in `reasoning` before either field is written.

   Instrument names are generated, not listed: a broad set of real issuers
   plus names invented from syllables with a random kind suffix. That is
   deliberate. A fixed pool of brands would teach the adapter those brands,
   and the next card would be the original failure again; generated names
   leave only the layout to learn. Use different seeds for `train.jsonl` and
   `eval.jsonl` — the invented names then differ between them, so eval
   accuracy on wallet samples measures whether the layout was learned.

2. **Train and export** following the toolkit's README (entry points and
   flags vary by toolkit release). Train against `train.jsonl`, evaluate on
   `eval.jsonl`, then export the `.fmadapter` package. Name the produced
   asset pack **`SpendingExtractionAdapter`** — that's the name
   `SpendingAgent` asks Background Assets for.

3. **Try it on a device** without any hosting setup: copy the `.fmadapter`
   into the app's `Application Support/Adapters/` directory on-device (Xcode
   → Devices → download/upload container, or the Files app for a
   development build). `SpendingAgent.developmentAdapterURL` picks up the
   first `.fmadapter` it finds there; `SpendingAgent.usesExtractionAdapter`
   tells you it loaded.

4. **Ship it** via Background Assets as an asset pack named
   `SpendingExtractionAdapter` (bundle with the `ba-package` tool, host via
   App Store Connect or your own server). Adapters are ~160 MB — do not
   bundle one inside the app binary. Part of that setup is adding the
   Background Assets keys (`BAHasManagedAssetPacks` etc.) to the app's
   Info.plist — `SpendingAgent` deliberately skips the Background Assets
   lookup until that key exists, because probing an unconfigured setup
   logs `BGSystemTaskSchedulerErrorDomain` errors on every launch.

## Keeping the data generator honest

The generator duplicates two strings from the app on purpose, so training and
inference distributions match. If you change either, update
`make_training_data.py` in the same commit:

- `SpendingAgent.extractionInstructions` and the prompt assembly in
  `SpendingAgent.extract` (mirrored in `INSTRUCTIONS` / `assemble_prompt`).
- `KnowledgeFact.promptLine` phrasing (mirrored in the knowledge-line
  templates).

Also keep the gold-answer conventions aligned with the `@Guide` descriptions
on `ReceiptExtraction`: grand total copied exactly as a plain decimal string,
zero-decimal currencies without decimals, dates never invented, category
titles copied verbatim into `category` from the existing list when one fits
and `proposedCategory` left empty — or, when none fits, `category` left
empty and the new title in `proposedCategory`. The existing categories are
rendered with the merchants filed under them (`category_prompt_line`, the
mirror of `CategoryFit.promptLine`). An adapter trained before the
`proposedCategory` slot existed was trained against a different output
shape and prompt, and has to be retrained from regenerated data.
