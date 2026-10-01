"""Stage 1 only: deterministic, conservative audit; no splitting or modelling."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

LABELS = ['negative', 'neutral', 'positive']
EXPECTED_SHA256 = 'ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e'


def normalise(text):
    """NFC, collapse Unicode whitespace to one ASCII space, strip ends only."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def duplicate_stats(df, key):
    groups = df.loc[df[key].ne('')].groupby(key, sort=False)
    sizes = groups.size()
    return {
        'groups': int(sizes.gt(1).sum()),
        'rows_in_groups': int(sizes[sizes.gt(1)].sum()),
        'excess_rows': int((sizes - 1).clip(lower=0).sum()),
        'conflicting_label_groups': int(groups.airline_sentiment.nunique().gt(1).sum()),
    }


def linked_groups(df):
    """Connect equal nonblank IDs or normalised text before removing any duplicate."""
    parent = {i: i for i in df.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for key in ['tweet_id', 'normalised_text']:
        first = {}
        for i, value in df[key].items():
            if not value:
                continue
            if value in first:
                a, b = find(i), find(first[value])
                parent[max(a, b)] = min(a, b)
            else:
                first[value] = i
    return pd.Series({i: find(i) for i in df.index}, dtype='int64')


def run_audit(root):
    root = Path(root)
    raw_path = root / 'data/original/Tweets.csv'
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, 'Original CSV changed; review provenance before auditing.'
    out = root / 'results/audit'
    out.mkdir(parents=True, exist_ok=True)
    # No automatic NA conversion: literal strings such as "NA" remain text.
    raw = pd.read_csv(raw_path, dtype='string', keep_default_na=False, encoding='utf-8')
    assert {'tweet_id', 'text', 'airline_sentiment'} <= set(raw.columns)
    df = raw.copy()
    df.insert(0, 'source_row', range(1, len(df) + 1))  # 1-based data record, not physical line
    df['normalised_text'] = df.text.map(normalise)
    missing = pd.DataFrame({
        'empty_cells': raw.eq('').sum(),
        'blank_including_whitespace': raw.apply(lambda x: x.str.strip().eq('').sum()),
    })
    missing.to_csv(out / 'missing_values.csv', index_label='column')
    initial = {
        'tweet_id': duplicate_stats(df, 'tweet_id'),
        'raw_text': duplicate_stats(df, 'text'),
        'normalised_text': duplicate_stats(df, 'normalised_text'),
        'blank_text': int(df.normalised_text.eq('').sum()),
        'invalid_label': int((~df.airline_sentiment.isin(LABELS)).sum()),
        'blank_tweet_id': int(df.tweet_id.str.strip().eq('').sum()),
        'non_decimal_nonblank_tweet_id': int((df.tweet_id.ne('') & ~df.tweet_id.str.fullmatch(r'[0-9]+')).sum()),
    }
    ledger, excluded = [], []

    def remove(mask, reason):
        nonlocal df
        before = len(df)
        gone = df.loc[mask].copy()
        gone['removal_reason'] = reason
        excluded.append(gone)
        df = df.loc[~mask].copy()
        ledger.append({'step': reason, 'before': before, 'removed': len(gone), 'remaining': len(df)})

    remove(df.normalised_text.eq(''), 'blank_or_missing_text')
    remove(~df.airline_sentiment.isin(LABELS), 'invalid_or_missing_label')
    df['linked_group'] = linked_groups(df)
    conflicts = df.groupby('linked_group').airline_sentiment.nunique()
    conflict_ids = conflicts[conflicts.gt(1)].index
    # Save every implicated row, including transitive links; never choose a majority label.
    df.loc[df.linked_group.isin(conflict_ids)].to_csv(out / 'conflicting_groups.csv', index=False)
    valid_duplicates = df.loc[df.tweet_id.duplicated(False) | df.normalised_text.duplicated(False)]
    valid_duplicates.to_csv(out / 'duplicate_groups.csv', index=False)
    remove(df.linked_group.isin(conflict_ids), 'quarantine_conflicting_label_components')
    # An ID should identify one text. Quarantine whole linked components if it does not.
    id_texts = df.loc[df.tweet_id.ne('')].groupby('tweet_id').normalised_text.nunique()
    inconsistent_ids = id_texts[id_texts.gt(1)].index
    inconsistent_components = df.loc[df.tweet_id.isin(inconsistent_ids), 'linked_group'].unique()
    remove(df.linked_group.isin(inconsistent_components), 'quarantine_inconsistent_id_text_components')
    remove(df.tweet_id.ne('') & df.tweet_id.duplicated(keep='first'), 'duplicate_tweet_id_keep_first_source_row')
    remove(df.normalised_text.duplicated(keep='first'), 'duplicate_normalised_text_keep_first_source_row')

    removed = pd.concat(excluded, ignore_index=True)
    assert removed.source_row.is_unique
    assert not set(removed.source_row) & set(df.source_row)
    assert len(raw) == len(removed) + len(df)
    assert df.normalised_text.is_unique
    assert df.loc[df.tweet_id.ne(''), 'tweet_id'].is_unique
    assert df.airline_sentiment.isin(LABELS).all() and df.normalised_text.ne('').all()
    assert df.tweet_id.map(lambda x: isinstance(x, str)).all()
    assert raw_path.read_bytes() and hashlib.sha256(raw_path.read_bytes()).hexdigest() == digest
    removed.to_csv(out / 'removed_rows.csv', index=False)
    pd.DataFrame(ledger).to_csv(out / 'removal_ledger.csv', index=False)
    df.to_csv(root / 'data/processed/tweets_audited.csv', index=False)
    distribution = pd.DataFrame({
        'starting': raw.airline_sentiment.value_counts().reindex(LABELS, fill_value=0),
        'final': df.airline_sentiment.value_counts().reindex(LABELS, fill_value=0),
    })
    distribution['final_percent'] = (100 * distribution['final'] / len(df)).round(4)
    distribution.to_csv(out / 'class_distribution.csv', index_label='sentiment')

    # Diagnostic alternatives only, NOT competing cleaning rules or extra removals.
    probe = raw.text.map(normalise)
    sensitivity = {
        'all_original_rows': len(raw),
        'unique_raw_tweet_id': int(raw.tweet_id.nunique()),
        'unique_raw_text': int(raw.text.nunique()),
        'unique_NFC_whitespace_text': int(probe.nunique()),
        'unique_NFC_whitespace_casefold_text': int(probe.str.casefold().nunique()),
        'chosen_policy_final': len(df),
        'previous_claim': 14250,
        'previous_claim_status': 'No prior notebook or rules provided. These diagnostics do not establish provenance for 14,250.',
    }
    pd.Series(sensitivity, name='value').to_csv(out / 'count_investigation.csv', index_label='diagnostic')
    confidence = pd.to_numeric(df.airline_sentiment_confidence, errors='coerce')
    summary = {
        'raw_sha256': digest, 'starting_count': len(raw), 'initial_checks': initial,
        'conflicting_components': len(conflict_ids),
        'removal_ledger': ledger, 'final_count': len(df),
        'at_least_10000': len(df) >= 10000,
        'class_distribution': distribution.reset_index(names='sentiment').to_dict(orient='records'),
        'count_investigation': sensitivity,
        'final_confidence_below_1': int(confidence.lt(1).sum()),
        'final_confidence_missing_or_nonnumeric': int(confidence.isna().sum()),
        'final_confidence_outside_0_1': int((confidence.lt(0) | confidence.gt(1)).sum()),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary, df, distribution
