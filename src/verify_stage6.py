"""Independent arithmetic/traceability verification of saved final test outputs.

No model inference, no training, no writes to original evaluation artifacts.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/stage6'; DATA=ROOT/'data/processed/stage2'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:pd.read_csv(p,dtype={'source_row':str,'tweet_id':str})
checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name
e=load(OUT/'notebook_execution.json'); protocol=load(ROOT/'results/stage5/frozen_test_protocol.json')
check('recorded_source_hashes',all(sha(ROOT/p)==h for p,h in e['source_hashes'].items()))
check('recorded_output_hashes',all(sha(OUT/p)==h for p,h in e['output_hashes'].items()))
preserved=load(OUT/'preservation.json')
check('earlier_artifacts_unchanged',preserved['before']==preserved['after'] and all(sha(ROOT/p)==h for p,h in preserved['before'].items()))
check('frozen_protocol_unchanged',sha(ROOT/'results/stage5/frozen_test_protocol.json')==load(OUT/'preflight.json')['protocol_sha256'])
check('six_frozen_checkpoint_hashes',len(protocol['checkpoints'])==6 and all(sha(ROOT/r['checkpoint'])==r['sha256'] for r in protocol['checkpoints']))
nbpath=ROOT/'notebooks/06_final_test_evaluation.ipynb'; nb=nbformat.read(nbpath,as_version=4)
check('notebook_hash',sha(nbpath)==e['notebook_sha256'])
cells=[c for c in nb.cells if c.cell_type=='code']
check('three_successful_code_cells',len(cells)==3 and all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells))
frame=read(DATA/'test.csv'); vocab=load(DATA/'vocabulary.json'); labels=['negative','neutral','positive']
mapping=load(DATA/'config.json')['label_to_id']; check('fixed_label_mapping',mapping==dict(zip(labels,range(3))))
table=pd.read_csv(OUT/'checkpoint_comparison.csv'); class_table=pd.read_csv(OUT/'per_class_metrics.csv')
test=np.load(DATA/'test_encoded.npz',allow_pickle=False)
check('test_source_label_alignment',test['source_rows'].tolist()==frame.source_row.tolist() and test['labels'].tolist()==frame.airline_sentiment.map(mapping).tolist())
expected_sample=[]
for r in protocol['checkpoints']:
    name=f"{r['model']}_seed{r['seed']}"; z=np.load(OUT/f'{name}_predictions.npz',allow_pickle=False)
    csv=read(OUT/f'{name}_predictions.csv'); report=load(OUT/f'{name}_metrics.json')
    y=z['labels']; p=z['predictions']; probs=z['probabilities']; logits=z['logits']
    check(name+'_source_labels',np.array_equal(y,test['labels']) and np.array_equal(z['source_rows'],test['source_rows']) and z['tweet_ids'].tolist()==frame.tweet_id.tolist())
    softmax=np.exp(logits-logits.max(1,keepdims=True)); softmax/=softmax.sum(1,keepdims=True)
    check(name+'_probabilities',np.isfinite(probs).all() and np.allclose(probs,softmax,atol=1e-7) and np.allclose(probs.sum(1),1,atol=1e-6) and np.array_equal(p,probs.argmax(1)))
    check(name+'_csv_traceability',csv.source_row.tolist()==frame.source_row.tolist() and csv.tweet_id.tolist()==frame.tweet_id.tolist() and np.array_equal(csv.true_label_id,y) and np.array_equal(csv.predicted_label_id,p) and np.allclose(csv[[f'probability_{c}' for c in labels]],probs,atol=1e-7))
    check(name+'_csv_label_names',csv.true_label.tolist()==[labels[i] for i in y] and csv.predicted_label.tolist()==[labels[i] for i in p])
    # Independent confusion-count arithmetic, without using the evaluation metric helper.
    cm=np.zeros((3,3),dtype=int); np.add.at(cm,(y,p),1)
    support=cm.sum(1); predicted=cm.sum(0); tp=cm.diagonal()
    precision=np.divide(tp,predicted,out=np.zeros(3),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros(3),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3),where=(precision+recall)!=0)
    values={'accuracy':float(tp.sum()/len(y)),'macro_f1':float(f1.mean()),'weighted_f1':float((f1*support).sum()/support.sum())}
    tab=table.loc[table.model.eq(r['model'])&table.seed.eq(r['seed'])].iloc[0]
    check(name+'_aggregate_metrics',all(np.isclose(v,report[k],atol=1e-12) and np.isclose(v,tab[k],atol=1e-12) for k,v in values.items()))
    check(name+'_confusion_matrix',cm.tolist()==report['confusion_matrix'] and support.tolist()==[1361,446,334])
    for i,cl in enumerate(labels):
        row=class_table.loc[class_table.model.eq(r['model'])&class_table.seed.eq(r['seed'])&class_table['class'].eq(cl)].iloc[0]
        check(name+'_'+cl+'_class_metrics',all(np.isclose(v,report['per_class'][cl][k],atol=1e-12) and np.isclose(v,row[k],atol=1e-12) for k,v in {'precision':precision[i],'recall':recall[i],'f1-score':f1[i],'support':support[i]}.items()))
    if r['seed']==42:
        for cl in labels:
            for correct in [True,False]:
                eligible=[str(row) for row,t,pr in zip(z['source_rows'],y,p) if labels[t]==cl and (t==pr)==correct]
                chosen=min(eligible,key=lambda row:hashlib.sha256(f'2026|{r["model"]}|{row}'.encode()).hexdigest())
                expected_sample.append((r['model'],chosen,cl,bool(correct)))
means=pd.read_csv(OUT/'model_summary.csv')
for kind in ['cnn','gru']:
    t=table.loc[table.model.eq(kind)]; row=means.loc[means.model.eq(kind)].iloc[0]
    check(kind+'_all_seeds_mean_sample_std',t.seed.tolist()==[42,43,44] and all(np.isclose(row[f'{m}_mean'],t[m].mean(),atol=1e-12) and np.isclose(row[f'{m}_sample_std'],t[m].std(ddof=1),atol=1e-12) for m in ['accuracy','macro_f1','weighted_f1']))
sample=read(OUT/'example_sample.csv')
check('reproducible_sample_exact',list(zip(sample.model,sample.source_row,sample.true_label,sample.correct))==expected_sample)
source=frame.set_index('source_row')
check('sample_text_and_unknowns',all(row.text==source.loc[row.source_row,'text'] and json.loads(row.unseen_tokens_json)==[t for t in json.loads(source.loc[row.source_row,'tokens_json']) if t not in vocab] for row in sample.itertuples()))
result={'all_passed':all(checks.values()),'checks':checks,'checked_checkpoint_count':6,'reproducible_example_count':len(sample),
        'scope':'saved-output arithmetic, identities, probability checks and sample reconstruction only; no rerun of inference',
        'original_execution':'complete, three cells, zero errors; no failed execution found in saved Stage 6 outputs'}
(OUT/'independent_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2)); print('Checks passed:',len(checks))
