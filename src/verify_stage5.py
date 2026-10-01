"""Independent saved-trial/selection verification and validation checkpoint reloads."""
import json
import numpy as np
import pandas as pd
import torch
from stage5 import ROOT, OUTER, BASE, earlier_hashes, sha, load_array, evaluate, fresh_model

torch.set_num_threads(2)
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
summary=load(OUTER/'summary.json'); plan=load(OUTER/'prespecified_plan.json')
assert load(OUTER/'preservation.json')['before']==earlier_hashes()
train,val=load_array('train'),load_array('validation')
weights=torch.tensor(len(train['labels'])/(3*np.bincount(train['labels'],minlength=3)),dtype=torch.float32)
all_runs=summary['grid_trials']+[r for runs in summary['seed_runs'].values() for r in runs if r['seed']!=42]
assert len(all_runs)==12 and sum(not r['reused_baseline'] for r in all_runs)==10
checks={}
for r in all_runs:
    kind=r['kind']; folder=ROOT/r['directory']; name=folder.name
    config=load(folder/'configuration.json'); h=pd.read_csv(folder/f'{kind}_history.csv')
    assert config['class_weights_train']==weights.tolist()
    assert config['seed']==r['seed'] and config['learning_rate']==r['learning_rate'] and config['dropout']==r['dropout']
    # Only lr/dropout/seed-derived fields can differ from the prespecified protocol.
    for k,v in plan['fixed_baseline_configuration'].items():
        if k not in ['seed','learning_rate','dropout','ordering','masking']: assert config[k]==v
    assert int(h.loc[h.validation_unmasked_macro_f1.idxmax(),'epoch'])==r['best_epoch']
    anchor=-1.; stale=0
    for row in h.itertuples():
        if row.validation_unmasked_macro_f1>anchor+.0001: anchor=row.validation_unmasked_macro_f1; stale=0
        else: stale+=1
        assert stale==row.patience_counter
    assert len(h)<=20 and (len(h)==20 or stale==3)
    p=folder/f'{kind}_best.pt'; checkpoint=torch.load(p,map_location='cpu',weights_only=True)
    assert checkpoint['settings']==config and checkpoint['epoch']==r['best_epoch']
    assert sha(p)==r['checkpoint_sha256']
    assert all(sha(ROOT/path)==digest for path,digest in checkpoint['references'].items())
    m=fresh_model(kind,checkpoint['vocabulary_size'],seed=999); m.dropout.p=config['dropout']
    m.load_state_dict(checkpoint['state_dict']); actual,scores=evaluate(m,val,weights)
    with np.load(folder/f'{kind}_best_validation.npz',allow_pickle=False) as z:
        assert np.array_equal(scores,z['logits']) and np.array_equal(scores.argmax(1),z['predictions'])
        assert np.array_equal(z['source_rows'],val['source_rows']) and np.array_equal(z['labels'],val['labels'])
    assert actual==r['validation']
    checks[name]=True
for kind in ['cnn','gru']:
    candidates=[r for r in summary['grid_trials'] if r['kind']==kind]
    choice=min(candidates,key=lambda r:(-r['validation']['macro_f1'],r['best_epoch'],r['grid_index']))
    assert choice==summary['selected'][kind]
    runs=summary['seed_runs'][kind]
    assert [r['seed'] for r in runs]==[42,43,44]
    assert all(r['learning_rate']==choice['learning_rate'] and r['dropout']==choice['dropout'] for r in runs)
    values=[r['validation']['macro_f1'] for r in runs]
    assert values==summary['variability'][kind]['validation_macro_f1']
    assert float(np.mean(values))==summary['variability'][kind]['mean']
    assert float(np.std(values,ddof=1))==summary['variability'][kind]['sample_std']
frozen=load(OUTER/'frozen_test_protocol.json')
assert len(frozen['checkpoints'])==6
for r in frozen['checkpoints']:
    assert sha(ROOT/r['checkpoint'])==r['sha256']
    expected=next(s for s in summary['seed_runs'][r['model']] if s['seed']==r['seed'])
    assert (ROOT/r['checkpoint']).parent==ROOT/expected['directory']
assert not summary['test_evaluated']
result={'all_passed':True,'trial_reload_checks':checks,'grid_trials':8,'fresh_runs':10,'reused_baselines':2,
        'selection_and_sample_std_verified':True,'six_final_checkpoints_frozen':True,'previous_outputs_preserved':True,
        'test_evaluated':False,'scope':'train class-count and validation inference verification; no additional training'}
(OUTER/'independent_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
