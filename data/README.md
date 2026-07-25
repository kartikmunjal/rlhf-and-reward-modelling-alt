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
