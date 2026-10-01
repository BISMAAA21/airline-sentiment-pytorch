# Reproducibility notes

The original experiment froze the data split, vocabulary, sequence-length cap, model-selection rule, tuning grid, and final test protocol before reading held-out test results.

The public copy preserves the source notebooks, stage scripts, selected result summaries, and plots. It deliberately omits the raw dataset, generated encoded arrays, checkpoints, predictions, and local Python environment. Those artifacts can be regenerated locally after obtaining the dataset.

The verified environment was Windows 11 x86-64 CPU, Python 3.13.7, and PyTorch 2.14.0+cpu. The final evaluation used the same 2,141 held-out examples for each seed. Seed 42, 43, and 44 results are summarized with sample standard deviation using ddof=1.

Re-running training can produce different wall-clock timings and may differ across hardware or library versions. Do not tune against the final test set.
