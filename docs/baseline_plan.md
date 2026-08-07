# Baseline plan

File layout and label schema below are confirmed (2026-08-07, both against a
small local sample and at full scale from inside `kernels/dev/dev_smoke_test.py`
— see CLAUDE.md rule 6 on why both checks matter).

## Decided sequencing (2026-08-07)

Two steps, in this order, not in parallel:

**Step 1 — v0 image baseline, trained ONLY on the 58 labeled rows.**
Purpose is exclusively to prove the *pipeline* end-to-end: DICOM loading ->
training loop -> checkpointing -> `kaggle kernels push`/run/pull -> a
correctly-formatted `submission.csv`. Its leaderboard score is expected to
be close to noise (58 examples across 12 targets is not enough to learn
anything real) and **must be treated as meaningless** — a "did the plumbing
work" signal, not "is this a good model." Do not spend time tuning it. Move
to Step 2 as soon as it runs clean end-to-end once.

**Step 2 — weak-labeling pipeline for the other 4349 report-only rows.**
This is the actual leverage point for a real score (see "label scarcity"
below for why). Starts immediately after Step 1 passes, not after Step 1 is
"good" — Step 1 will never be good on its own. Since `test.csv` has no
`Report` column, weak-labeling only ever grows the *training* set; final
inference stays image-only regardless of how training labels were sourced,
so Step 1's inference path (offline, image-only) doesn't need to change
when Step 2 lands — only the training data feeding it does.

The rest of this document describes Step 1 in detail (the "v0" sections
below — "v0" names the model version, "Step 1" names the process phase),
then Step 2 under "Step 2: after Step 1 passes end-to-end once."

## IMPORTANT: label scarcity - read before designing v1+

Confirmed at full scale (4407 train rows):
- Only **58 / 4407 (1.3%)** train rows have *any* non-null structured target
  label. The other 4349 rows have a `Report` but no structured labels at all.
- The `Report` column is present for **100%** of train rows (mean length
  ~1100 chars; confirmed multilingual - Spanish/Dutch/German text seen).
- `test.csv` has **only** `StudyInstanceUID` - no `Report` column, and only
  3 rows in the current public copy.

This matches RSNA's own framing of the 2026 challenge as the first to
require learning from real-world reports "where findings are complex and
answers are not neatly organized in a table" — the 58 structured-label rows
read like a small gold/calibration set, not the primary supervision signal.
Practical implications:
- A supervised image classifier trained only on the 58 labeled rows (v0
  below) is a fine **pipeline validation** step but will not have enough
  data to learn anything beyond noise — treat its leaderboard score as
  "did the plumbing work," not "is this a good model."
- The likely real path to a good score is **weak-labeling the other 4349
  rows from their `Report` text** (keyword/rule-based per language, or a
  small classifier trained on the 58 gold rows) before training the image
  model on the enlarged pseudo-labeled set. This moves the report-text work
  from "v2 nice-to-have" to something worth prioritizing much sooner than
  originally planned below.
- Because `test.csv` has no `Report` column, the report text can only be a
  **training-time** signal (weak labeling) — final inference is still
  necessarily image-only, so the "Inference" section below stays valid
  regardless of how the label-scarcity problem gets solved. Re-verify this
  assumption against the full/private test set if it becomes available
  later in the competition, since the public `test.csv` seen so far may not
  reflect final scoring data.

## Why unimodal-image-first for Step 1

- Fewer moving parts than image+text fusion -> faster to get to a valid,
  scored submission and prove the pipeline.
- Establishes and tests the two things `CLAUDE.md` mandates before any real
  GPU spend: checkpoint-to-disk during training, and a fully offline
  inference notebook. Cheaper to debug those on a small model than on the
  Step-2-trained one.
- Since `test.csv` has no `Report` column, image-only inference is not a
  Step-1-only simplification — it's the permanent shape of the final
  submission (see "Decided sequencing"). Step 2 changes what trains the
  model, not what the model consumes at inference time.

## Data handling (Step 1, deliberately simple)

- **One slice per study**, not the full 3D volume. Studies have multiple
  series (confirmed: `train_series.csv`/`test_series.csv` give
  `Anatomical_Plane` [Sagittal/Coronal/Axial], `Fluid_Sensitive` [0/1], and
  `Fat_Suppression` [0/1] per `SeriesInstanceUID`) and many slices each; for
  v0, pick a single representative slice per study to keep this a plain 2D
  image classification problem:
  - Prefer a series with `Anatomical_Plane == "Sagittal"` and
    `Fluid_Sensitive == 1` — most relevant to ACL/meniscus/cartilage/effusion
    findings, and directly available as a column rather than needing to
    parse DICOM `SeriesDescription` text.
  - Within that series, take the **middle slice by index** as a first pass
    (cheap, no heuristic needed). A pixel-intensity-variance-based "most
    informative slice" heuristic is a fast v0.1 improvement if the middle
    slice underperforms.
- Confirmed from a sample DICOM header: `Modality=MR`, `Rows=Columns=640`,
  `PixelSpacing~=[0.234, 0.234]mm`. Resize 640x640 -> 224x224, single-channel
  -> replicate to 3-channel if using an ImageNet-pretrained backbone,
  normalize with ImageNet stats as a starting point (revisit once pixel
  value ranges post `RescaleSlope`/`RescaleIntercept`/windowing are checked).
- No augmentation beyond horizontal flip + light rotation/brightness jitter
  for v0 — keep training fast and debuggable.

## Model

- Backbone: `resnet18` (torchvision, ImageNet-pretrained) — small, fast,
  well-understood. Swap for `efficientnet_b0` later if time/compute allows;
  not worth the extra tuning burden for v0.
- Head: replace final FC with a single `Linear(in_features, 12)` — one
  sigmoid logit per target. Multi-label, not multi-class (labels are not
  mutually exclusive).
- Loss: `BCEWithLogitsLoss`, optionally with `pos_weight` per target computed
  from the class balance found in exploration (targets are likely imbalanced
  — e.g. Fracture rare, Effusion more common).

## Training

- **Training set is exactly the 58 rows with a non-null structured label**
  — do not include the other 4349 report-only rows in Step 1 (that's the
  whole point of Step 2). 58 rows across 12 targets is too small for a
  meaningful held-out split to mean anything; still exercise the
  **GroupKFold by StudyInstanceUID** split code path (2 folds is plenty)
  so the mechanism is proven, but don't read anything into the resulting
  val AUC — see "Decided sequencing" above.
- Budget: 3-5 epochs, batch size 32, image size 224, AdamW, lr 3e-4 with
  cosine or step decay, mixed precision (`torch.cuda.amp`) to fit Kaggle's
  GPU quota. Should train in well under a minute on 58 rows — this step is
  about proving the mechanics run, not about compute budget.
- Metric: macro-averaged `roc_auc_score` (sklearn, `average="macro"`)
  computed on the val fold each epoch — wire this up correctly now (it's
  the actual competition metric) even though the Step 1 number itself is
  meaningless; Step 2 is where this metric starts mattering.
- **Checkpointing (per `CLAUDE.md` rule 1):** save model + optimizer state
  to `outputs/checkpoints/` after every epoch, plus a separate `best.pt`
  whenever val macro-AUC improves. Training script must be resumable from
  the last checkpoint if a Kaggle session gets killed mid-run. Prove this
  works now, on the cheap 58-row run, rather than discovering a
  checkpointing bug during a longer Step 2 run.

## Inference (separate, offline notebook)

- Per `CLAUDE.md` rule 2/4: the scored notebook only loads `best.pt` (from a
  Kaggle Dataset holding the checkpoint, attached to the notebook), runs the
  same single-slice-per-study preprocessing on the test set, and writes
  `submission.csv`. No `pip install`, no downloads, no training code.
- Sanity-check column names/order/row-count/no-NaNs the same way
  `src/inference/make_trivial_submission.py` already does — reuse that
  validation logic rather than re-deriving it.
- At well under an hour of inference for a single 2D CNN forward pass, this
  leaves enormous headroom under the 9h runtime cap — that headroom is for
  the Step-2-trained model later, not Step 1 itself.

## Explicit non-goals for Step 1

- No training on the 4349 report-only rows (that's Step 2, not Step 1).
- No report text / multimodal fusion.
- No 3D volume modeling (multi-slice/multi-series aggregation).
- No test-time augmentation or ensembling.
- No hyperparameter search.
- No tuning for leaderboard score — Step 1's score is expected to be noise
  and that's fine; see "Decided sequencing" above.

Each of these is a reasonable later improvement once Step 2 has produced a
baseline whose score is actually worth improving.

## Step 2: after Step 1 passes end-to-end once

Weak-labeling moved ahead of multimodal fusion — fusion doesn't matter much
if there's only 58 labeled examples to fuse with:

1. **Weak-label the 4349 report-only rows.** Start with per-language
   keyword/rule matching against the `Report` text for each of the 12
   targets (the 58 gold rows are a natural small validation set for
   precision/recall of the rules before trusting them). A small text
   classifier trained on the 58 gold rows is a fallback if rules underfit.
   This is the highest-leverage next step — it's what actually grows the
   usable training set.
2. Retrain the v0 image model on the enlarged pseudo-labeled set; compare
   macro-AUC against the 58-row v0 baseline to confirm weak labels actually
   help before investing further.
3. Add multi-series/multi-slice aggregation (e.g. 2.5D: stack a few adjacent
   slices as channels, or per-series feature pooling) if it beats the
   single-slice baseline.
4. Only then consider heavier backbones, k-fold ensembling, or TTA — once
   there's a working, weakly-supervised baseline to spend the remaining
   GPU-hour budget on. Note: since `test.csv` has no `Report` column (see
   above), there's no image+text late-fusion step at *inference* time — the
   report text's only role is training-time weak labeling.
