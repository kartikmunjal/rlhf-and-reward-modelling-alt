# Data cards

## Jigsaw Toxic Comment Classification

Download Kaggle's `train.csv` into `data/raw/jigsaw/train.csv`. The raw file is
ignored by Git. The pipeline validates its schema and creates an 80/10/10 split
by SHA-256 hashing the immutable comment ID with seed 2025. This makes row
reordering irrelevant and prevents calibration or test examples from entering
training.

The locked `jigsaw-jd-v1` target mapping is:

- `hate_harassment`: `toxic OR severe_toxic OR insult OR identity_hate`
- `sexualized`: `obscene`
- `harmful_violent`: `threat`

This is an operational proxy mapping, not a claim that toxicity and hate speech
are identical constructs. In particular, Jigsaw has no clean sexual-content
label; `obscene` is only the closest available proxy.

## Adjacent-benign v1

`adjacent_benign_v1.csv` contains 60 synthetic, hand-curated, all-negative
stress cases: 20 reclaimed/quoted-language examples, 20 clinical/medical
examples, and 20 news-reporting-on-violence examples.

These examples were written for this repository and are not sampled from the
Jigsaw test set. They measure a narrow failure mode: whether contextually benign
text is flagged by any safety category. They are a stress test, not a population
estimate and not evidence of demographic parity. Text was frozen before model
evaluation; changes require a new versioned file.

## Safety v2 external data

V2 data are downloaded and normalized only by
`scripts/prepare_safety_v2_data.py`. The script resolves or reuses exact
Hugging Face repository revisions and writes raw-file plus normalized-file
hashes to `data/processed/safety_v2/data_manifest.json`.

BeaverTails 330k repeats prompt-response pairs as annotation rows. Per
preregistration amendment 1, the adapter aggregates each mapped target by
strict majority and excludes a pair when any mapped target is tied. The
manifest records the full aggregation audit; normalized BeaverTails rows are
therefore unique pair-level examples rather than raw annotations.

Required sources:

- `PKU-Alignment/BeaverTails`, CC BY-NC 4.0. This non-commercial license means
  the v2 adapter and derived training use are research/job-portfolio artifacts,
  not commercially deployable assets without separate permission.
- `toxigen/toxigen-data`, human-annotated configuration. Access may require
  accepting the dataset's access terms and authenticating to Hugging Face.
- `Paul/hatecheck`, CC BY 4.0.
- The original Jigsaw Unintended Bias/Civil Comments CSV containing identity
  columns. The reduced `google/civil_comments` Hugging Face conversion omits
  those columns and is intentionally rejected by the adapter.

External test sets are never concatenated into training. Normalized data and
their manifest remain local because they may contain harmful text and because
source licenses differ.
