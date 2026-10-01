"""Independent saved-output verification; does not import the Stage 2 builder."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage2'
DATA=ROOT/'data/processed/stage2'
def read(p): return pd.read_csv(p,dtype='string',keep_default_na=False)
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name

raw=read(ROOT/'data/processed/tweets_audited.csv')
kept=read(DATA/'tweets_distinct.csv')
removed=read(OUT/'removed_rows.csv')
all_rows=pd.concat([kept,removed]).sort_values('source_row',key=lambda s:s.astype(int))
check('one_outcome_per_stage1_row',all_rows.source_row.is_unique and set(all_rows.source_row)==set(raw.source_row))
original=raw.set_index('source_row')
check('all_original_columns_preserved',all(all_rows[c].tolist()==original.loc[all_rows.source_row,c].tolist()
      for c in raw.columns if c!='source_row'))
check('distinct_minimum',len(kept)==kept.tokens_json.nunique() and len(kept)>=10000)
check('valid_nonempty_tokens',all(isinstance(json.loads(t),list) and len(json.loads(t))>0 for t in kept.tokens_json))
g=all_rows.groupby('tokens_json')
bad_keys=set(g.airline_sentiment.nunique().loc[lambda s:s>1].index)
expected_conflicts=set(all_rows.loc[all_rows.tokens_json.isin(bad_keys),'source_row'])
check('entire_conflicting_groups_quarantined',expected_conflicts==set(removed.loc[removed.removal_reason.eq('quarantine_conflicting_token_group'),'source_row']))
agree=all_rows.loc[~all_rows.tokens_json.isin(bad_keys)]
expected_dupes=set(agree.loc[agree.tokens_json.duplicated(keep='first'),'source_row'])
check('earliest_representative_policy',expected_dupes==set(removed.loc[removed.removal_reason.eq('duplicate_token_sequence_keep_first_source_row'),'source_row']))
check('no_unrecorded_exclusions',len(raw)-len(kept)==len(removed)==127)
ledger=read(OUT/'removal_ledger.csv')
running=len(raw)
for r in ledger.itertuples():
    check('ledger_'+r.step,int(r.before)==running and int(r.removed)==int(removed.removal_reason.eq(r.step).sum())
          and int(r.remaining)==running-int(r.removed))
    running=int(r.remaining)
manifest=read(DATA/'split_manifest.csv')
check('one_shared_split_partition',manifest.source_row.is_unique and set(manifest.source_row)==set(kept.source_row))
frames={n:read(DATA/f'{n}.csv') for n in ['train','validation','test']}
for n,f in frames.items():
    check(n+'_manifest_matches',set(f.source_row)==set(manifest.loc[manifest.split.eq(n),'source_row']))
    check(n+'_full_tokens_unique',f.tokens_json.is_unique)
check('full_tokens_disjoint',sum(f.tokens_json.nunique() for f in frames.values())==kept.tokens_json.nunique())
for field in ['source_row','tweet_id','text','normalised_text','tokens_json']:
    partitions=[set(f[field]) for f in frames.values()]
    check(field+'_cross_split_disjoint',not any(partitions[i]&partitions[j] for i in range(3) for j in range(i+1,3)))
config=load(DATA/'config.json')
vocab=load(DATA/'vocabulary.json')
train=frames['train']
counts=Counter(t for ts in train.tokens_json for t in json.loads(ts))
special=['<PAD>','<UNK>','<URL>','<USER>']
expected_words=sorted([t for t in counts if t not in special],key=lambda t:(-counts[t],t))
check('vocabulary_exact_training_only_derivation',vocab=={t:i for i,t in enumerate(special+expected_words)})
check('fit_rows_only_train',set(config['fit_source_rows'])==set(train.source_row))
lengths=np.array([len(json.loads(t)) for t in train.tokens_json])
length_table=pd.read_csv(OUT/'training_length_comparison.csv')
for q,r in zip([.5,.9,.95,.99,1.],length_table.itertuples()):
    check(r.candidate+'_training_length',int(r.length)==int(np.quantile(lengths,q,method='higher')))
    check(r.candidate+'_discard_count',int(r.discarded_tokens)==int(np.maximum(lengths-r.length,0).sum()))
passing=length_table.loc[(length_table.truncated_tweets_percent<=1)&(length_table.discarded_tokens_percent<=.1)]
check('training_only_length_selection',config['length_cap']==int(passing.length.min()))
collisions=defaultdict(list)
for n,f in frames.items():
    z=np.load(DATA/f'{n}_encoded.npz',allow_pickle=False)
    saved_ids=z['input_ids']
    saved_lengths=z['lengths']
    check(n+'_source_alignment',z['source_rows'].tolist()==f.source_row.tolist())
    check(n+'_labels_alignment',z['labels'].tolist()==[config['label_to_id'][s] for s in f.airline_sentiment])
    expected=[[vocab.get(t,1) for t in json.loads(ts)[:config['length_cap']]] for ts in f.tokens_json]
    check(n+'_encoded_inputs_exact',all(saved_ids[i,:len(seq)].tolist()==seq and not saved_ids[i,len(seq):].any()
          and saved_lengths[i]==len(seq) for i,seq in enumerate(expected)))
    for row,seq in zip(f.source_row,expected): collisions[tuple(seq)].append((n,row))
groups=[v for v in collisions.values() if len(v)>1]
cross=[g for g in groups if len({n for n,_ in g})>1]
actual_collision_rows={(n,row) for g in groups for n,row in g}
export=read(OUT/'encoded_collision_rows.csv')
check('collision_diagnostic_exact',actual_collision_rows==set(zip(export.split,export.source_row)))
summary=load(OUT/'summary.json')
check('collision_diagnostics_honestly_recorded',len(cross)==summary['cross_split_encoded_collision_groups']
      and len(groups)==summary['encoded_diagnostics']['encoded_collision_groups']
      and len(actual_collision_rows)==summary['encoded_collision_rows'])
tokens_by_row={row:json.loads(ts) for row,ts in zip(kept.source_row,kept.tokens_json)}
cause_count=0
for group in groups:
    sequences=[tokens_by_row[row] for _,row in group]
    if all(len(ts)<=config['length_cap'] for ts in sequences):
        differing=[pos for pos in range(len(sequences[0])) if len({ts[pos] for ts in sequences})>1]
        # Every differing position must consist solely of words absent from training.
        cause_count += bool(differing) and all(all(ts[pos] not in vocab for ts in sequences) for pos in differing)
diagnostics=load(OUT/'encoded_diagnostics.json')
check('unseen_word_cause_independently_verified',cause_count==diagnostics['groups_explained_by_unseen_tokens_without_truncation'])
check('training_has_no_unknowns',all(t in vocab for t in counts))
check('hard_integrity_checks_pass',summary['all_checks_passed'] and all(load(OUT/'checks.json').values()))
frozen=load(ROOT/'config/stage2_frozen_inputs.json')
check('frozen_inputs_unchanged',all(sha(ROOT/p)==h for p,h in frozen['hashes'].items()) and config['length_cap']==frozen['length_cap'])
preserve=load(OUT/'stage1_preservation.json')
check('stage1_still_unchanged',all(sha(ROOT/p)==h for p,h in preserve['before'].items()) and preserve['identical'])
check('materials_unchanged',all(sha(ROOT/r['file'])==r['sha256'] for r in load(ROOT/'results/materials_review/extracted.json')))
result={'verification_checks':checks,'structural_verification_passed':all(checks.values()),
        'distinct_full_token_rows':len(kept),'distinct_encoded_inputs':len(collisions),
        'encoded_collision_groups':len(groups),'encoded_collision_rows':len(actual_collision_rows),
        'cross_split_encoded_collision_groups':len(cross),
        'groups_with_only_unseen_word_differences':cause_count,
        'data_preparation_verified':all(checks.values()),
        'status':'verified; model implementation and training not performed'}
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
