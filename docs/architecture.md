# Architecture

## Pipeline

```text
raw CSV
  -> audit and normalization
  -> duplicate/conflict handling
  -> frozen stratified split
  -> training-only vocabulary
  -> padded token IDs and true lengths
  -> CNN or packed GRU
  -> three-class logits
  -> accuracy, macro-F1, weighted-F1, and error analysis
```

## Shared design

Both models consume the same encoded inputs, vocabulary, split manifest, sequence-length cap, class labels, and evaluation protocol. This keeps the architecture comparison focused on the model family rather than different data preparation.

## CNN path

The CNN embeds token IDs, applies a width-three one-dimensional convolution, masks padded positions before max pooling, applies dropout, and projects the pooled representation to three sentiment logits.

## GRU path

The GRU embeds token IDs, packs sequences using their true lengths, processes them with a single recurrent layer, takes the final hidden state, applies dropout, and projects it to three sentiment logits.
