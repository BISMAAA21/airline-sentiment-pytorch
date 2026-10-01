"""Verify baseline evidence and independently reload checkpoints on validation only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stage4 import ROOT, OUT, DATA, sha, preserved, load_array, evaluate
from models import fresh_model

torch.set_num_threads(2)
summary=json.loads((OUT/'summary.json').read_text())
configuration=json.loads((OUT/'configuration.json').read_text())
preservation=json.loads((OUT/'preservation.json').read_text())
assert preservation['before']==preserved()==preservation['after']
train,val=load_array('train'),load_array('validation')
counts=np.bincount(train['labels'],minlength=3)
weights=torch.tensor(len(train['labels'])/(3*counts),dtype=torch.float32)
assert weights.tolist()==configuration['class_weights_train']
checks={'previous_stages_unchanged':True,'class_weights_training_only':True}
for kind in ['cnn','gru']:
    history=pd.read_csv(OUT/f'{kind}_history.csv')
    best=int(history.loc[history.validation_unmasked_macro_f1.idxmax(),'epoch'])
    checkpoint=torch.load(OUT/f'{kind}_best.pt',map_location='cpu',weights_only=True)
    assert best==checkpoint['epoch']==summary['models'][kind]['best_epoch']
    assert sha(OUT/f'{kind}_best.pt')==summary['models'][kind]['checkpoint_sha256']
    assert all(sha(ROOT/p)==h for p,h in checkpoint['references'].items())
    model=fresh_model(kind,checkpoint['vocabulary_size'],seed=123)
    model.load_state_dict(checkpoint['state_dict'])
    actual,scores=evaluate(model,val,weights)
    saved=np.load(OUT/f'{kind}_best_validation.npz',allow_pickle=False)
    assert np.array_equal(saved['source_rows'],val['source_rows'])
    assert np.array_equal(saved['labels'],val['labels'])
    assert np.array_equal(scores,saved['logits']) and np.array_equal(scores.argmax(1),saved['predictions'])
    assert actual==summary['models'][kind]['validation']
    anchor=-1.; stale=0
    for row in history.itertuples():
        if row.validation_unmasked_macro_f1>anchor+.0001: anchor=row.validation_unmasked_macro_f1; stale=0
        else: stale+=1
        assert stale==row.patience_counter
    assert len(history)<=20 and (len(history)==20 or stale==3)
    checks[kind+'_checkpoint_reload_and_selection']=True
    checks[kind+'_early_stopping_verified']=True
result={'checks':checks,'all_passed':all(checks.values()),'test_evaluated':False,'tuning_started':False,
        'scope':'saved artifacts, training class counts and validation checkpoint reload only'}
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
