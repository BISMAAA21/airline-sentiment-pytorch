"""Custom small sentiment models. No pretrained assets or global training state."""
import hashlib
import regex
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def validate_inputs(ids, lengths):
    if ids.ndim != 2 or lengths.shape != (ids.shape[0],):
        raise ValueError('Expected IDs [batch,time] and lengths [batch]')
    if ids.shape[0] == 0 or torch.any(lengths < 1) or torch.any(lengths > ids.shape[1]):
        raise ValueError('Lengths must be positive and fit the input width')
    mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths.to(ids.device)[:, None]
    if torch.any(ids[~mask] != 0) or torch.any(ids[mask] == 0):
        raise ValueError('Only right padding may use PAD=0')
    return mask


class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(embedding_dim, hidden_size, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, ids, lengths):
        mask = validate_inputs(ids, lengths)
        features = torch.relu(self.conv(self.embedding(ids).transpose(1, 2)))
        pooled = features.masked_fill(~mask[:, None, :], float('-inf')).amax(dim=2)
        return self.head(self.dropout(pooled))


class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, ids, lengths):
        validate_inputs(ids, lengths)
        packed = pack_padded_sequence(self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(self.dropout(hidden[-1]))


def fresh_model(kind, vocab_size, seed=42):
    """Fresh random parameters every call; never reuse diagnostic weights.

    Embedding is constructed first in both classes, giving the same initial table
    at the same seed. Defaults initialise weights randomly; PAD stays zero.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return {'cnn': SentimentCNN, 'gru': SentimentGRU}[kind](vocab_size)


class TrainingWordMasker:
    """Stateless reproducible 2% word masking, explicitly enabled only for train.

    Hash key = seed|epoch|source_row|position. Identical tokens receive equivalent
    treatment across architectures, batch order, and batch sizes. No labels used.
    """
    def __init__(self, vocabulary, seed=42, probability=0.02):
        self.seed, self.probability = int(seed), float(probability)
        if not 0 <= self.probability <= 1:
            raise ValueError('Invalid probability')
        self.eligible = torch.zeros(len(vocabulary), dtype=torch.bool)
        word = regex.compile(r"#?[\p{L}\p{M}\p{N}_]+(?:['\u2019][\p{L}\p{M}\p{N}_]+)*")
        negations = {'no', 'not', 'never', 'neither', 'nor', 'without', 'cannot'}
        for token, index in vocabulary.items():
            low = token.casefold().replace('\u2019', "'")
            protected = low.lstrip('#') in negations or low.endswith("n't") or index < 4
            self.eligible[index] = bool(word.fullmatch(token)) and not protected

    def __call__(self, ids, source_rows, epoch, *, split, training):
        result = ids.clone()
        changed = torch.zeros_like(ids, dtype=torch.bool)
        if not training or split != 'train':
            return result, changed
        if len(source_rows) != len(ids):
            raise ValueError('Source rows must align with the batch')
        eligible = self.eligible.to(ids.device)[ids]
        for row, position in eligible.nonzero().tolist():
            key = f'{self.seed}|{epoch}|{source_rows[row]}|{position}'.encode('utf-8')
            uniform = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'big') / 2**64
            if uniform < self.probability:
                changed[row, position] = True
        result[changed] = 1
        return result, changed
