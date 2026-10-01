"""Stage 2 data preparation only. No model construction, training or test scoring."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd
import regex
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/stage2'
DATA = ROOT / 'data/processed/stage2'
LABELS = {'negative': 0, 'neutral': 1, 'positive': 2}
SPECIAL = ['<PAD>', '<UNK>', '<URL>', '<USER>']
# URL candidates stop at whitespace or HTML delimiters. Terminal prose punctuation
# is split off and retained. URL-internal punctuation and path/query text are replaced.
URL = regex.compile(r'https?://[^\s<>]+|www\.[^\s<>]+', regex.IGNORECASE)
MENTION = regex.compile(r'(?<![\p{L}\p{N}_@])@[A-Za-z0-9_]+')
TOKENS = regex.compile(r'<URL>|<USER>|#?[\p{L}\p{M}\p{N}_]+(?:[\x27\u2019][\p{L}\p{M}\p{N}_]+)*|\X')
NEGATIONS = {'no','not','never','neither','nor','without','cannot'}


def tokenise(text):
    text = unicodedata.normalize('NFC', text)
    text = regex.sub(r'\s+', ' ', text).strip()
    def url_marker(match):
        value = match.group(0)
        stripped = value.rstrip('.,!?;:)]}')
        return ' <URL> ' + value[len(stripped):]
    text = URL.sub(url_marker, text)
    text = MENTION.sub(' <USER> ', text)
    return [t for t in TOKENS.findall(text) if not t.isspace()]


def token_key(tokens):
    return json.dumps(tokens, ensure_ascii=False, separators=(',', ':'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def preservation_manifest():
    paths = [ROOT/'data/original/Tweets.csv', ROOT/'data/original/twitter-airline-sentiment.zip',
             ROOT/'data/processed/tweets_audited.csv', ROOT/'notebooks/01_dataset_audit.ipynb',
             ROOT/'src/audit.py', ROOT/'results/notebook_verification.json',
             *sorted((ROOT/'results/audit').glob('*'))]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    before_hashes = preservation_manifest()
    frozen = json.loads((ROOT/'config/stage2_frozen_inputs.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in frozen['hashes'].items()), 'Frozen inputs changed'
    assert before_hashes['data\\original\\Tweets.csv'] == 'ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e'
    raw = pd.read_csv(ROOT/'data/processed/tweets_audited.csv', dtype='string', keep_default_na=False)
    assert len(raw) == 14398 and raw.source_row.is_unique and raw.tweet_id.is_unique
    work = raw.copy()
    work['tokens_json'] = work.text.map(lambda text: token_key(tokenise(text)))
    work['token_count'] = work.tokens_json.map(lambda text: len(json.loads(text)))
    work['sequence_group'] = work.tokens_json.map(lambda text: hashlib.sha256(text.encode('utf-8')).hexdigest())
    assert work.groupby('sequence_group').tokens_json.nunique().le(1).all()
    initial_unique = int(work.tokens_json.nunique())
    groups = work.groupby('tokens_json', sort=False)
    sizes = groups.size()
    initial_group_counts = {'repeated_groups': int(sizes.gt(1).sum()),
                            'rows_in_repeated_groups': int(sizes[sizes.gt(1)].sum())}
    duplicates = work.loc[work.tokens_json.duplicated(False)]
    duplicates.to_csv(OUT/'repeated_token_rows.csv', index=False)
    ledger, excluded = [], []
    def remove(mask, reason):
        nonlocal work
        count_before = len(work)
        gone = work.loc[mask].copy()
        gone['removal_reason'] = reason
        excluded.append(gone)
        work = work.loc[~mask].copy()
        ledger.append({'step':reason,'before':count_before,'removed':len(gone),'remaining':len(work)})
    remove(work.token_count.eq(0), 'empty_token_sequence')
    remove(~work.airline_sentiment.isin(LABELS), 'invalid_label')
    conflict_keys = work.groupby('tokens_json').airline_sentiment.nunique()
    conflict_keys = conflict_keys[conflict_keys.gt(1)].index
    conflicts = work.loc[work.tokens_json.isin(conflict_keys)]
    conflicts.to_csv(OUT/'conflicting_token_rows.csv',index=False)
    remove(work.tokens_json.isin(conflict_keys), 'quarantine_conflicting_token_group')
    # Original source order is preserved by Stage 1. Earliest source record wins only
    # when all labels agree; source_row is the 1-based original CSV data record.
    work = work.sort_values('source_row', key=lambda s:s.astype(int), kind='stable')
    remove(work.tokens_json.duplicated(keep='first'), 'duplicate_token_sequence_keep_first_source_row')
    removals = pd.concat(excluded, ignore_index=True)
    assert removals.source_row.is_unique and len(raw) == len(removals) + len(work)
    assert work.tokens_json.is_unique
    membership = raw[['source_row','tweet_id']].merge(
        pd.concat([work[['source_row']].assign(outcome='retained'),
                   removals[['source_row','removal_reason']].rename(columns={'removal_reason':'outcome'})]),
        on='source_row',validate='one_to_one')
    membership.to_csv(OUT/'source_outcomes.csv',index=False)
    removals.to_csv(OUT/'removed_rows.csv',index=False)
    pd.DataFrame(ledger).to_csv(OUT/'removal_ledger.csv',index=False)
    work.to_csv(DATA/'tweets_distinct.csv',index=False)
    summary = {'stage1_rows':len(raw),'stage1_unique_normalised_texts':int(raw.normalised_text.nunique()),
               'initial_unique_token_sequences':initial_unique, **initial_group_counts,
               'conflicting_groups':len(conflict_keys),'conflicting_rows':len(conflicts),
               'removal_ledger':ledger,'distinct_rows_before_split':len(work),
               'distinct_token_sequences_before_split':int(work.tokens_json.nunique()),
               'minimum_10000_passed':len(work)>=10000}
    save_json(OUT/'summary.json',summary)
    assert len(work)>=10000, 'STOP: fewer than 10,000 distinct usable examples. No split created.'

    # Reuse the reviewed assignments, never regenerate or search for a new split.
    manifest = pd.read_csv(DATA/'split_manifest.csv', dtype='string', keep_default_na=False)
    assert manifest.source_row.is_unique and set(manifest.source_row)==set(work.source_row)
    split_frames = {name:work.loc[work.source_row.isin(manifest.loc[manifest.split.eq(name),'source_row'])].sort_index()
                    for name in ['train','validation','test']}
    for name, frame in split_frames.items():
        saved = pd.read_csv(DATA/f'{name}.csv', dtype='string', keep_default_na=False)
        assert frame.astype('string').reset_index(drop=True).equals(saved), 'Frozen split content differs'
    train = split_frames['train']
    token_lists = [json.loads(t) for t in train.tokens_json]
    counts = Counter(t for tokens in token_lists for t in tokens)
    retained = sorted([t for t in counts if t not in SPECIAL],
                      key=lambda t:(-counts[t],t))
    vocab = {t:i for i,t in enumerate(SPECIAL+retained)}
    save_json(DATA/'vocabulary.json',vocab)
    pd.DataFrame([{'token':t,'train_frequency':c,'in_vocabulary':t in vocab}
                  for t,c in sorted(counts.items(),key=lambda item:(-item[1],item[0]))]).to_csv(OUT/'training_token_counts.csv',index=False)
    lengths = np.array([len(t) for t in token_lists])
    candidates = []
    for label, q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99),('maximum',1.)]:
        cap = int(np.quantile(lengths,q,method='higher'))
        discarded = np.maximum(lengths-cap,0)
        padded = np.maximum(cap-lengths,0)
        candidates.append({'candidate':label,'length':cap,'truncated_tweets':int((lengths>cap).sum()),
                           'truncated_tweets_percent':float(100*(lengths>cap).mean()),
                           'discarded_tokens':int(discarded.sum()),
                           'discarded_tokens_percent':float(100*discarded.sum()/lengths.sum()),
                           'fixed_padding_tokens':int(padded.sum()),
                           'fixed_padding_percent':float(100*padded.sum()/(len(lengths)*cap)),
                           'fixed_batch_token_slots':int(len(lengths)*cap)})
    # Predeclared training-only rule: at most 1% of training tweets and 0.1%
    # of training tokens truncated. Choose smallest passing candidate, else maximum.
    selected = min([c for c in candidates if c['truncated_tweets_percent']<=1.0
                    and c['discarded_tokens_percent']<=.1],key=lambda c:c['length'])
    cap = frozen['length_cap']
    assert selected['length'] == cap, 'Training length evidence changed; review required'
    pd.DataFrame(candidates).to_csv(OUT/'training_length_comparison.csv',index=False)
    pd.DataFrame({'source_row':train.source_row,'token_count':lengths}).to_csv(OUT/'training_lengths.csv',index=False)
    class_counts = train.airline_sentiment.value_counts()
    optional_weights = {label:len(train)/(3*int(class_counts[label])) for label in LABELS}
    config = {'seed':42,'split_design':[.70,.15,.15],'label_to_id':LABELS,'vocabulary_size':len(vocab),
              'vocabulary_min_train_frequency':1,'observed_negation_exception':False,
              'observed_training_token_types':len(counts),'split_policy':'reuse frozen manifest',
              'special_ids':{t:vocab[t] for t in SPECIAL},'length_cap':cap,
              'length_selection_rule':'smallest p50/p90/p95/p99/max with <=1% training tweets and <=0.1% training tokens truncated',
              'length_candidate_selected':selected['candidate'],'truncation':'right; retain first length_cap tokens',
              'padding':'right; zero PAD; dynamic batch padding up to cap',
              'optional_training_class_weights':optional_weights,
              'weighted_loss_status':'design option only; no loss or model constructed',
              'fit_source_rows':train.source_row.tolist(),
              'pretrained_assets':False,'models_built':False}
    save_json(DATA/'config.json',config)
    checks = {'disjoint_source_ids':len(set().union(*(set(f.source_row) for f in split_frames.values())))==len(work),
              'vocabulary_training_only':set(vocab)-set(SPECIAL)<=set(counts),
              'all_required_training_tokens_in_vocab':set(counts)<=set(vocab),
              'full_sequence_uniqueness':work.tokens_json.is_unique}
    encoded_keys = defaultdict(list)
    encoding_summary = {}
    distributions = []
    for name, frame in split_frames.items():
        all_tokens = [json.loads(t) for t in frame.tokens_json]
        seqs = [[vocab.get(t,vocab['<UNK>']) for t in ts[:cap]] for ts in all_tokens]
        true_lengths = np.array([len(seq) for seq in seqs],dtype=np.int64)
        ids = np.zeros((len(seqs),cap),dtype=np.int64)
        for j,seq in enumerate(seqs):
            ids[j,:len(seq)] = seq
            encoded_keys[tuple(seq)].append((name,str(frame.iloc[j].source_row)))
        y = np.array([LABELS[label] for label in frame.airline_sentiment],dtype=np.int64)
        np.savez_compressed(DATA/f'{name}_encoded.npz',input_ids=ids,lengths=true_lengths,labels=y,
                            source_rows=frame.source_row.to_numpy(dtype=str))
        # Structural checks only. No test examples, length distributions, OOV rates or scores are displayed.
        checks[f'{name}_padding_valid'] = bool(np.all((ids!=0)==(np.arange(cap)[None,:]<true_lengths[:,None])))
        checks[f'{name}_ids_in_range'] = bool(ids.min()>=0 and ids.max()<len(vocab))
        if name == 'train':
            checks['no_training_unknown_tokens'] = bool(not np.any(ids==vocab['<UNK>']))
        encoding_summary[name]={'rows':len(frame),'distinct_sequences':int(frame.tokens_json.nunique())}
        for label in LABELS:
            distributions.append({'split':name,'sentiment':label,'rows':int(frame.airline_sentiment.eq(label).sum())})
    collisions = [{'source_row':row,'split':name,'collision_group':i,'group_rows':len(entries)}
                  for i,(seq,entries) in enumerate(encoded_keys.items()) if len(entries)>1 for name,row in entries]
    pd.DataFrame(collisions,columns=['source_row','split','collision_group','group_rows']).to_csv(OUT/'encoded_collision_rows.csv',index=False)
    cross = [entries for entries in encoded_keys.values() if len({name for name,_ in entries})>1]
    collision_groups = [entries for entries in encoded_keys.values() if len(entries)>1]
    by_row = work.set_index('source_row').tokens_json
    collision_tokens = [[json.loads(by_row.loc[row]) for _,row in g] for g in collision_groups]
    diagnostics = {'encoded_collision_groups':len(collision_groups),
                   'encoded_collision_rows':len(collisions),'distinct_encoded_inputs':len(encoded_keys),
                   'cross_split_encoded_collision_groups':len(cross),
                   'training_only_groups':sum(all(n=='train' for n,_ in g) for g in collision_groups),
                   'groups_involving_held_out':sum(any(n!='train' for n,_ in g) for g in collision_groups),
                   'rows_by_split':{n:sum(name==n for g in collision_groups for name,_ in g) for n in split_frames},
                   'colliding_rows_truncated':sum(len(ts)>cap for group in collision_tokens for ts in group),
                   'groups_explained_by_unseen_tokens_without_truncation':sum(
                       all(len(ts)<=cap for ts in group) and len({token_key(ts) for ts in group})==len(group)
                       and any(t not in vocab for ts in group for t in ts) for group in collision_tokens),
                   'policy':'Non-fatal representation diagnostic; no exclusions, relabelling, refitting or test scoring.'}
    save_json(OUT/'encoded_diagnostics.json',diagnostics)
    for field in ['tweet_id','sequence_group','tokens_json']:
        sets = [set(f[field]) for f in split_frames.values()]
        checks[f'no_cross_split_{field}'] = not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    checks['stage1_preserved'] = before_hashes == preservation_manifest()
    checks['frozen_split_preprocessing_and_length_outputs_preserved'] = all(sha(ROOT/p)==h for p,h in frozen['hashes'].items())
    save_json(OUT/'stage1_preservation.json',{'before':before_hashes,'after':preservation_manifest(),
                                          'identical':checks['stage1_preserved']})
    save_json(OUT/'checks.json',checks)
    pd.DataFrame(distributions).to_csv(OUT/'split_class_distribution.csv',index=False)
    summary.update({'splits':encoding_summary,'vocabulary_size':len(vocab),'selected_length':cap,
                    'length_comparison':candidates,'encoded_collision_rows':len(collisions),
                    'cross_split_encoded_collision_groups':len(cross),'all_checks_passed':all(checks.values()),
                    'observed_training_token_types':len(counts),
                    'encoded_diagnostics':diagnostics,
                    'status':'complete' if all(checks.values()) else 'STOP: integrity failure; review required'})
    save_json(OUT/'summary.json',summary)
    return summary


def make_batch(input_ids, lengths):
    """Model-free helper: right padding, true lengths and a mask shared by both models."""
    lengths = torch.as_tensor(lengths,dtype=torch.long)
    width = int(lengths.max())
    ids = torch.as_tensor(input_ids,dtype=torch.long)[:,:width]
    mask = torch.arange(width)[None,:] < lengths[:,None]
    assert torch.all(ids[~mask]==0)
    return {'input_ids':ids,'lengths':lengths,'attention_mask':mask}
