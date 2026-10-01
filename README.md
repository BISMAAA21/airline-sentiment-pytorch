# Airline Sentiment with PyTorch

Reproducible experiments comparing a custom one-dimensional CNN and a GRU for three-class airline sentiment classification.

This repository is a curated public version of a larger experimental workspace. It keeps the model, preprocessing, training, tuning, evaluation, and reproducibility work while excluding local environments, raw data, checkpoints, reports, and review artifacts.

## Problem

Airline-service tweets are classified as **negative**, **neutral**, or **positive**. The project investigates whether two small models trained from scratch can learn useful sentiment representations under the same frozen preprocessing and data split.

## Dataset

The experiments use the [Twitter US Airline Sentiment dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment).

The raw dataset is intentionally not committed. See [`data/README.md`](data/README.md) for provenance and local placement instructions.

The verified experiment retained 14,398 rows after the audit and produced 14,271 distinct examples after preprocessing and duplicate/conflict handling:

| Split | Examples |
| --- | ---: |
| Train | 9,989 |
| Validation | 2,141 |
| Test | 2,141 |

The vocabulary was fitted from training data only and contained 13,898 IDs, including reserved markers. The shared sequence-length cap was 46 tokens.

## Preprocessing

The pipeline uses the same decisions for both models:

1. Audit invalid, empty, conflicting, and duplicate records.
2. Normalize Unicode and whitespace while preserving case, punctuation, emoji, negation, and contractions.
3. Replace URLs and mentions with explicit markers.
4. Tokenize into word/number/underscore runs and Unicode grapheme clusters.
5. Keep one representative for agreeing duplicate token sequences and quarantine conflicting groups.
6. Fit the vocabulary on training examples only.
7. Encode sequences with right padding and explicit lengths.
8. Apply deterministic 2% word masking during training only.

## Models

### Custom CNN

- 64-dimensional trainable embedding
- one-dimensional convolution with 64 output channels, kernel size 3, and padding 1
- ReLU activation
- masked global max pooling over valid tokens
- dropout
- linear three-class prediction head

### GRU

- 64-dimensional trainable embedding
- one-layer GRU with hidden size 64
- packed sequences using true sequence lengths
- dropout
- linear three-class prediction head

Neither model uses pretrained weights. Parameters are initialized from scratch for each run.

## Experiment methodology

- PyTorch on CPU
- Adam optimization
- training-weighted cross-entropy loss
- batch size 32
- gradient clipping at 1.0
- maximum of 20 epochs
- early stopping patience of 3 epochs
- validation macro-F1 used for checkpoint selection
- a fixed hyperparameter grid for learning rate and dropout
- three selected-configuration seeds: 42, 43, and 44
- the test set was accessed only after the configuration and checkpoint protocol was frozen

The selected validation configurations were:

| Model | Learning rate | Dropout |
| --- | ---: | ---: |
| CNN | 0.001 | 0.3 |
| GRU | 0.0005 | 0.5 |

## Evaluation metrics

The project reports accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrices, and seed variability. Mean and sample standard deviation are calculated over the three seed-specific runs on the same held-out test examples; they are not confidence intervals.

## Verified results

The table below is copied from the saved Stage 6 summaries in this project. It reports mean ± sample standard deviation across seeds 42, 43, and 44.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | ---: | ---: | ---: |
| CNN | 0.764752 ± 0.005914 | 0.708842 ± 0.003018 | 0.768672 ± 0.002926 |
| GRU | 0.754009 ± 0.009912 | 0.696298 ± 0.005709 | 0.759337 ± 0.007154 |

Each run evaluated the same 2,141 held-out examples. The CNN has the higher mean score on this fixed split, but the experiment does not establish general superiority or statistical significance.

### Selected evidence

- [Baseline comparison](results/stage4/baseline_comparison.csv)
- [Baseline learning curves](results/stage4/learning_curves.png)
- [Tuning comparison](results/stage5/tuning_comparison.csv)
- [Seed comparison](results/stage5/seed_comparison.csv)
- [Test model summary](results/stage6/model_summary.csv)
- [Per-class test metrics](results/stage6/per_class_metrics.csv)
- [Test model comparison](results/stage6/test_model_comparison.png)
- [Test confusion matrices](results/stage6/test_confusion_matrices.png)
- [Validation/test comparison](results/stage6/validation_test_comparison.png)

## Reproducibility

The original experiments were verified on Windows 11 x86-64 CPU with Python 3.13.7 and PyTorch 2.14.0+cpu. The public copy does not include the original virtual environment, raw dataset, checkpoints, or generated arrays.

To recreate a local environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-torch.lock.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-stage2.lock.txt
```

Download the dataset described in [`data/README.md`](data/README.md), place the CSV at `data/original/Tweets.csv`, and execute the notebooks in order:

1. `01_dataset_audit.ipynb`
2. `02_preprocessing_and_split.ipynb`
3. `03_model_verification.ipynb`
4. `04_baseline_training.ipynb`
5. `05_tuning_and_seed_variability.ipynb`
6. `06_final_test_evaluation.ipynb`

The stage runners in `src/` use project-relative paths and regenerate ignored `data/processed/` and `results/` artifacts. Review the corresponding notebook and protocol before running an experiment; training and evaluation are not required merely to inspect the saved public evidence.

## Limitations

- The dataset is historical and domain-specific; performance may not transfer to current airline or general social-media text.
- The test set is one fixed split, and only three training seeds were evaluated.
- Class imbalance affects the relationship between accuracy, weighted-F1, and macro-F1.
- Near-duplicates, annotation noise, conversational dependence, truncation at 46 tokens, and unseen-word representation collisions remain limitations.
- The results are descriptive evidence from this experiment, not a production claim or a significance test.

## Repository layout

```text
src/                 preprocessing, models, training, tuning, evaluation, verification
notebooks/            six executed experiment workflows
results/              selected summaries, tables, and plots
data/original/        provenance only; raw data is downloaded locally
docs/                 architecture, reproducibility, and security notes
```

## Security and data handling

No API keys, credentials, passwords, private keys, local environment files, virtual environments, raw Git metadata, or model checkpoints belong in the public repository. See [`docs/security.md`](docs/security.md).
