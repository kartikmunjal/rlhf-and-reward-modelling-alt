# Safety classifier v2 runbook

Read `docs/safety_v2_preregistered_plan.md` first. The plan and ordered ledger
are locked. The commands below implement them; they do not authorize changing
the trial matrix.

## 1. Prepare external data

Obtain the original Jigsaw Unintended Bias `train.csv` with identity columns.
Then run:

```bash
python scripts/prepare_safety_v2_data.py \
  --civil-comments-csv data/raw/civil_comments/train.csv \
  --output-dir data/processed/safety_v2
```

The first successful invocation resolves current source commits and writes
`data_manifest.json`. Preserve that file. Any repeat must pin those revisions:

```bash
python scripts/prepare_safety_v2_data.py \
  --civil-comments-csv data/raw/civil_comments/train.csv \
  --output-dir data/processed/safety_v2_rebuild \
  --revision-lock data/processed/safety_v2/data_manifest.json
```

Do not continue if normalized hashes differ. ToxiGen may require an approved
Hugging Face account and `HF_TOKEN`.

## 2. Smoke validation

Smoke runs validate plumbing only and are never research trials:

```bash
python scripts/train_safety_v2.py \
  --trial-id unweighted_bce_seed2025 \
  --max-train-examples 512 \
  --max-eval-examples 128 \
  --output-root checkpoints/safety_v2_smoke
```

The smoke output must not be placed under the production checkpoint root.

## 3. Run all 12 trials

On the RTX 3070:

```bash
python scripts/run_safety_v2_matrix.py --resume
```

The runner executes the ledger in order and writes
`results/safety_v2/matrix_status.json` after every state transition. A failed
trial remains recorded and stops the matrix. Fixing an implementation bug does
not erase that record; restarting uses `--resume`.

On Windows, invoke the runner through Task Scheduler as done for v1 so SSH
disconnects cannot terminate it. Expect v2 to take materially longer than v1:
the training set includes roughly 301k additional BeaverTails pairs and the
external evaluation includes Civil Comments.

## 4. Aggregate only after completion

```bash
python scripts/aggregate_safety_v2.py
```

Without `--allow-incomplete`, aggregation refuses to select a model until all
12 trial metrics exist. `--allow-incomplete` is diagnostic only and its output
is not publishable.

## 5. Blinded error analysis

Export deterministic errors from the selected trial:

```bash
python scripts/safety_error_analysis.py export \
  --predictions results/safety_v2/TRIAL_ID/predictions.npz \
  --output results/safety_v2/error_analysis_v1/annotation_sheet.csv \
  --version safety-v2-errors-v1
```

Two annotators independently complete the `_a` and `_b` columns. Only then:

```bash
python scripts/safety_error_analysis.py analyze \
  --annotations results/safety_v2/error_analysis_v1/annotation_sheet.csv \
  --output results/safety_v2/error_analysis_v1/analysis.json \
  --n-trials 12
```

The analyzer refuses incomplete sheets and reports agreement rather than
inventing consensus.

## 6. Generate model card and demo

```bash
python scripts/generate_safety_model_card.py

python scripts/safety_inference_demo.py \
  --text "Example text" \
  --model-dir checkpoints/safety_v2/TRIAL_ID \
  --metrics results/safety_v2/TRIAL_ID/metrics.json
```

The demo hashes rather than stores the input and prints category probabilities,
frozen thresholds, decisions, and a high-impact-use warning.

## Artifact policy

Commit compact generated reports, aggregate metrics, the model card, and data
manifests only when licenses permit. Never commit raw Jigsaw/Civil Comments
data, harmful-text evaluation exports, model weights, optimizer states, or
unblinded annotation sheets.
