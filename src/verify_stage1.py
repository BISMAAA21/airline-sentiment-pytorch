"""Independent Stage 1 review using csv + graph traversal, not audit.py helpers."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with (ROOT / path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def key(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def groups(rows, field):
    result = defaultdict(list)
    for row in rows:
        if row[field]:
            result[row[field]].append(row)
    return list(result.values())


def stats(rows, field):
    all_groups = groups(rows, field)
    duplicates = [g for g in all_groups if len(g) > 1]
    conflicts = [g for g in all_groups if len({r['airline_sentiment'] for r in g}) > 1]
    return {'duplicate_groups': len(duplicates), 'duplicate_group_rows': sum(map(len, duplicates)),
            'conflicting_groups': len(conflicts), 'conflicting_group_rows': sum(map(len, conflicts))}


raw = read('data/original/Tweets.csv')
clean = read('data/processed/tweets_audited.csv')
removed = read('results/audit/removed_rows.csv')
conflict_export = read('results/audit/conflicting_groups.csv')
duplicate_export = read('results/audit/duplicate_groups.csv')
ledger = read('results/audit/removal_ledger.csv')
summary = json.loads((ROOT / 'results/audit/summary.json').read_text(encoding='utf-8'))
provenance = json.loads((ROOT / 'docs/dataset_provenance.json').read_text(encoding='utf-8'))
checks = {}


def check(name, value):
    checks[name] = bool(value)
    assert value, name


digest = hashlib.sha256((ROOT / 'data/original/Tweets.csv').read_bytes()).hexdigest()
check('original_csv_checksum', digest == provenance['files']['Tweets.csv']['sha256'] == summary['raw_sha256'])
with zipfile.ZipFile(ROOT / 'data/original/twitter-airline-sentiment.zip') as archive:
    check('original_csv_matches_archive_bytes', archive.read('Tweets.csv') == (ROOT / 'data/original/Tweets.csv').read_bytes())
check('archive_checksum', hashlib.sha256((ROOT / 'data/original/twitter-airline-sentiment.zip').read_bytes()).hexdigest()
      == provenance['files']['twitter-airline-sentiment.zip']['sha256'])
original_columns = list(raw[0])
for i, row in enumerate(raw, 1):
    row['source_row'] = str(i)
    row['normalised_text'] = key(row['text'])
by_source = {r['source_row']: r for r in raw}
check('saved_original_columns_unchanged', all(r[c] == by_source[r['source_row']][c]
      for r in clean + removed for c in original_columns))
check('normalised_keys_match_rule', all(r['normalised_text'] == key(r['text']) for r in clean + removed))
check('unique_source_records', len({r['source_row'] for r in clean + removed}) == len(raw) == len(clean) + len(removed))
check('complete_partition', {r['source_row'] for r in clean + removed} == set(by_source))

# Independently recreate connected groups with an adjacency list and depth-first search.
valid = [r for r in raw if r['normalised_text'] and r['airline_sentiment'] in {'negative','neutral','positive'}]
adj = {r['source_row']: set() for r in valid}
for field in ['tweet_id', 'normalised_text']:
    for g in groups(valid, field):
        anchor = g[0]['source_row']
        for row in g[1:]:
            node = row['source_row']
            adj[anchor].add(node)
            adj[node].add(anchor)
components, seen = [], set()
for start in adj:
    if start in seen:
        continue
    stack, component = [start], set()
    seen.add(start)
    while stack:
        node = stack.pop()
        component.add(node)
        for neighbour in adj[node] - seen:
            seen.add(neighbour)
            stack.append(neighbour)
    components.append(component)
conflicts = [g for g in components if len({by_source[i]['airline_sentiment'] for i in g}) > 1]
conflict_rows = set().union(*conflicts) if conflicts else set()
check('conflict_export_exact_membership', conflict_rows == {r['source_row'] for r in conflict_export})
check('conflict_export_component_membership', {frozenset(r['source_row'] for r in g) for g in groups(conflict_export, 'linked_group')}
      == {frozenset(g) for g in conflicts})
expected_duplicate_rows = set()
for field in ['tweet_id', 'normalised_text']:
    for g in groups(valid, field):
        if len(g) > 1:
            expected_duplicate_rows.update(r['source_row'] for r in g)
check('duplicate_export_exact_membership', expected_duplicate_rows == {r['source_row'] for r in duplicate_export})

remaining = list(raw)
expected_reasons, expected_ledger = {}, []


def exclude(ids, reason):
    global remaining
    before = len(remaining)
    gone = [r for r in remaining if r['source_row'] in ids]
    expected_reasons.update({r['source_row']: reason for r in gone})
    remaining = [r for r in remaining if r['source_row'] not in ids]
    expected_ledger.append({'step': reason, 'before': before, 'removed': len(gone), 'remaining': len(remaining)})


exclude({r['source_row'] for r in remaining if not r['normalised_text']}, 'blank_or_missing_text')
exclude({r['source_row'] for r in remaining if r['airline_sentiment'] not in {'negative','neutral','positive'}}, 'invalid_or_missing_label')
exclude(conflict_rows, 'quarantine_conflicting_label_components')
inconsistent = {r['source_row'] for g in groups(remaining, 'tweet_id')
                if len({r['normalised_text'] for r in g}) > 1 for r in g}
bad_components = set().union(*(g for g in components if g & inconsistent)) if inconsistent else set()
exclude(bad_components, 'quarantine_inconsistent_id_text_components')
for field, reason in [('tweet_id', 'duplicate_tweet_id_keep_first_source_row'),
                      ('normalised_text', 'duplicate_normalised_text_keep_first_source_row')]:
    exclude({r['source_row'] for g in groups(remaining, field) for r in g[1:]}, reason)
check('exact_retained_records_and_order', [r['source_row'] for r in remaining] == [r['source_row'] for r in clean])
check('exact_removal_reasons', expected_reasons == {r['source_row']: r['removal_reason'] for r in removed})
check('sequential_ledger', expected_ledger == [{k: v if k == 'step' else int(v) for k,v in r.items()} for r in ledger]
      == summary['removal_ledger'])
final_stats = {field: stats(clean, field) for field in ['tweet_id','normalised_text']}
check('no_remaining_exact_duplicates_or_conflicts', all(value == 0 for result in final_stats.values() for value in result.values()))
check('valid_text_and_labels', all(r['normalised_text'] and r['airline_sentiment'] in {'negative','neutral','positive'} for r in clean))
distribution = Counter(r['airline_sentiment'] for r in clean)
check('class_distribution', dict(distribution) == {r['sentiment']: int(r['final']) for r in read('results/audit/class_distribution.csv')})
check('counts', len(raw) == 14640 and len(clean) == 14398 and len(removed) == 242)
check('conflict_summary_count', len(conflicts) == summary['conflicting_components'] == 26 and len(conflict_rows) == 59)
verification = json.loads((ROOT / 'results/notebook_verification.json').read_text(encoding='utf-8'))
check('saved_artifact_hashes', all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
      for p, sha in verification['artifact_sha256'].items()))
nb_path = ROOT / 'notebooks/01_dataset_audit.ipynb'
check('saved_notebook_hash', hashlib.sha256(nb_path.read_bytes()).hexdigest() == verification['notebook_sha256'])
nb = json.loads(nb_path.read_text(encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
check('all_notebook_cells_executed_without_error', [c['execution_count'] for c in code_cells] == list(range(1,len(code_cells)+1))
      and not any(o['output_type'] == 'error' for c in code_cells for o in c['outputs']))
result = {'verified_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
          'method': 'Independent csv parsing and graph traversal; no audit.py imports.',
          'initial_groups': {field: stats(raw, field) for field in ['tweet_id','normalised_text']},
          'conflicting_connected_groups': len(conflicts), 'conflicting_rows': len(conflict_rows),
          'final_checks': final_stats, 'ledger': expected_ledger, 'final_class_counts': dict(distribution),
          'original_csv_sha256': digest,
          'materials_files': [str(p.relative_to(ROOT)) for p in (ROOT/'materials').rglob('*') if p.is_file()]}
(ROOT / 'results/stage1_review_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
