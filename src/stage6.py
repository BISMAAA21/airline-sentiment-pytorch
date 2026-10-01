"""Final frozen-checkpoint evaluation. No training, masking, selection or tuning."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from models import fresh_model

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/stage6'; DATA=ROOT/'data/processed/stage2'
CLASSES=['negative','neutral','positive']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def read(p):return pd.read_csv(p,dtype='string',keep_default_na=False)

def preflight():
    # Verify frozen records before ANY test forward pass.
    original=load(ROOT/'results/stage5/final_manifest.json')
    assert all(sha(ROOT/p)==h for p,h in original.items()), 'Stage 5 frozen artifact changed'
    protocol=load(ROOT/'results/stage5/frozen_test_protocol.json')
    assert len(protocol['checkpoints'])==6
    assert {(r['model'],r['seed']) for r in protocol['checkpoints']}=={(m,s) for m in ['cnn','gru'] for s in [42,43,44]}
    refs={**protocol['data_references'],protocol['test_file_reference']['path']:protocol['test_file_reference']['sha256']}
    refs.update(load(ROOT/'config/stage2_frozen_inputs.json')['hashes'])
    refs.update(load(ROOT/'results/stage5/preservation.json')['before'])
    refs.update(load(ROOT/'results/stage2/stage1_preservation.json')['before'])
    assert all(sha(ROOT/p)==h for p,h in refs.items()), 'Frozen data/earlier artifact changed'
    config=load(DATA/'config.json'); vocab=load(DATA/'vocabulary.json')
    assert config['label_to_id']==dict(zip(CLASSES,range(3))) and config['length_cap']==46
    assert config['special_ids']=={'<PAD>':0,'<UNK>':1,'<URL>':2,'<USER>':3} and len(vocab)==13898
    frames={name:read(DATA/f'{name}.csv') for name in ['train','validation','test']}
    manifest=read(DATA/'split_manifest.csv')
    assert [len(f) for f in frames.values()]==[9989,2141,2141]
    for name,f in frames.items():
        assigned=manifest.loc[manifest.split.eq(name)].set_index('source_row')
        assert f.source_row.is_unique and set(f.source_row)==set(assigned.index)
        assert f.airline_sentiment.tolist()==assigned.loc[f.source_row,'airline_sentiment'].tolist()
        assert f.tweet_id.tolist()==assigned.loc[f.source_row,'tweet_id'].tolist()
    for field in ['source_row','tweet_id','tokens_json']:
        groups=[set(f[field]) for f in frames.values()]
        assert not any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    with np.load(DATA/'test_encoded.npz',allow_pickle=False) as z: test={k:z[k].copy() for k in z.files}
    f=frames['test']
    assert test['source_rows'].tolist()==f.source_row.tolist()
    assert test['labels'].tolist()==[config['label_to_id'][s] for s in f.airline_sentiment]
    for i,ts in enumerate(f.tokens_json):
        seq=[vocab.get(t,1) for t in json.loads(ts)[:46]]
        assert test['input_ids'][i,:len(seq)].tolist()==seq and not test['input_ids'][i,len(seq):].any()
        assert test['lengths'][i]==len(seq)
    for r in protocol['checkpoints']:
        p=ROOT/r['checkpoint']; assert sha(p)==r['sha256']
        c=torch.load(p,map_location='cpu',weights_only=True)
        assert c['kind']==r['model'] and c['epoch']==r['best_validation_epoch'] and c['vocabulary_size']==len(vocab)
        assert all(c['settings'][k]==r[k] for k in ['seed','learning_rate','dropout'])
        assert c['settings']['length_cap']==46 and all(sha(ROOT/p)==h for p,h in c['references'].items())
    protected={**original,**refs,**{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'materials').rglob('*') if p.is_file()}}
    save('preflight.json',{'all_passed':True,'checkpoint_count':6,'test_rows':2141,'label_mapping':config['label_to_id'],
                          'test_class_counts':np.bincount(test['labels'],minlength=3).tolist(),
                          'protocol_sha256':sha(ROOT/'results/stage5/frozen_test_protocol.json'),
                          'checks':'frozen manifests, all six checkpoint identities/settings, split/labels, vocabulary, cap, exact test encodings',
                          'example_sampling':'seed42 only, one per model x true class x correct/incorrect bucket, smallest SHA256 of 2026|model|source_row; no keyword search'})
    return protocol,test,f,vocab,protected

def metric_report(y,p):
    return {'accuracy':float(accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,p,labels=[0,1,2],average='weighted',zero_division=0)),
            'per_class':classification_report(y,p,labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist()}

def run():
    OUT.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    protocol,test,frame,vocab,before=preflight()
    reports=[]; sample_frames=[]; class_rows=[]
    for r in protocol['checkpoints']:
        name=f"{r['model']}_seed{r['seed']}"; tick=time.perf_counter()
        c=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=True)
        model=fresh_model(r['model'],len(vocab),seed=0); model.dropout.p=r['dropout']
        model.load_state_dict(c['state_dict'],strict=True); model.eval()
        assert not model.training and not model.dropout.training
        before_state={k:v.clone() for k,v in model.state_dict().items()}
        parts=[]
        with torch.inference_mode():
            for start in range(0,len(frame),32):
                lengths=torch.tensor(test['lengths'][start:start+32],dtype=torch.long)
                ids=torch.tensor(test['input_ids'][start:start+32,:int(lengths.max())],dtype=torch.long)
                # No masker is imported/called. These are the original saved test IDs.
                parts.append(model(ids,lengths).numpy().copy())
        logits=np.concatenate(parts); probabilities=torch.softmax(torch.tensor(logits),dim=1).numpy()
        prediction=logits.argmax(1); truth=test['labels']
        assert np.isfinite(logits).all() and np.allclose(probabilities.sum(1),1,atol=1e-6)
        assert all(torch.equal(before_state[k],v) for k,v in model.state_dict().items())
        np.savez_compressed(OUT/f'{name}_predictions.npz',logits=logits,probabilities=probabilities,
                            predictions=prediction,labels=truth,source_rows=test['source_rows'],tweet_ids=frame.tweet_id.to_numpy(dtype=str))
        export=pd.DataFrame({'source_row':test['source_rows'],'tweet_id':frame.tweet_id,'true_label_id':truth,
                             'true_label':[CLASSES[i] for i in truth],'predicted_label_id':prediction,
                             'predicted_label':[CLASSES[i] for i in prediction],
                             **{f'probability_{c}':probabilities[:,i] for i,c in enumerate(CLASSES)}})
        export.to_csv(OUT/f'{name}_predictions.csv',index=False)
        report=metric_report(truth,prediction)
        # Recalculate from persisted numeric outputs, not in-memory model scores.
        with np.load(OUT/f'{name}_predictions.npz',allow_pickle=False) as z:
            assert metric_report(z['labels'],z['predictions'])==report
            assert np.array_equal(z['probabilities'].argmax(1),z['predictions'])
        report.update(model=r['model'],seed=r['seed'],checkpoint=r['checkpoint'],checkpoint_sha256=r['sha256'],
                      evaluation_seconds=time.perf_counter()-tick,parameters_unchanged=True,saved_metrics_verified=True)
        save(f'{name}_metrics.json',report); reports.append(report)
        for cl in CLASSES:class_rows.append({'model':r['model'],'seed':r['seed'],'class':cl,**report['per_class'][cl]})
        if r['seed']==42:
            joined=export.merge(frame[['source_row','text','tokens_json']],on='source_row',validate='one_to_one')
            joined['correct']=prediction==truth
            joined['sample_rank']=joined.source_row.map(lambda row:hashlib.sha256(f'2026|{r["model"]}|{row}'.encode()).hexdigest())
            joined['model']=r['model']; joined['seed']=42
            for cl in CLASSES:
                for correct in [True,False]:
                    pool=joined.loc[joined.true_label.eq(cl)&joined.correct.eq(correct)].sort_values('sample_rank')
                    chosen=pool.head(1).copy()
                    chosen['unseen_tokens_json']=chosen.tokens_json.map(lambda ts:json.dumps([t for t in json.loads(ts) if t not in vocab],ensure_ascii=False))
                    sample_frames.append(chosen)
        print(name, 'test accuracy',report['accuracy'],'macro-F1',report['macro_f1'],flush=True)
        del model,c,before_state
    pd.concat(sample_frames,ignore_index=True).to_csv(OUT/'example_sample.csv',index=False)
    pd.DataFrame(class_rows).to_csv(OUT/'per_class_metrics.csv',index=False)
    table=pd.DataFrame([{k:r[k] for k in ['model','seed','accuracy','macro_f1','weighted_f1','evaluation_seconds']} for r in reports])
    table.to_csv(OUT/'checkpoint_comparison.csv',index=False)
    aggregated=[]
    for kind in ['cnn','gru']:
        rows=table.loc[table.model.eq(kind)]
        aggregated.append({'model':kind,**{f'{metric}_{stat}':float(rows[metric].mean() if stat=='mean' else rows[metric].std(ddof=1))
                           for metric in ['accuracy','macro_f1','weighted_f1'] for stat in ['mean','sample_std']}})
    pd.DataFrame(aggregated).to_csv(OUT/'model_summary.csv',index=False)
    per_class=pd.DataFrame(class_rows)
    per_class.groupby(['model','class'])[['precision','recall','f1-score','support']].agg(['mean','std']).to_csv(OUT/'per_class_seed_summary.csv')
    summary={'checkpoints':reports,'model_summary':aggregated,'class_order':CLASSES,'test_rows_per_seed':len(frame),
             'sample_sd_ddof':1,'aggregation':'mean/sample SD of three seed metrics; no pooled predictions or ensemble',
             'all_saved_metrics_verified':True,'no_training_or_tuning':True,'previous_artifacts_preserved':all(sha(ROOT/p)==h for p,h in before.items()),
             'evaluation_wall_seconds_before_plots':time.perf_counter()-started}
    assert summary['previous_artifacts_preserved']
    save('preservation.json',{'before':before,'after':{p:sha(ROOT/p) for p in before},'identical':True})
    save('summary.json',summary)
    plots(reports,table)
    return summary

def plots(reports,table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for ax,r in zip(axes.flat,reports):
        cm=np.array(r['confusion_matrix']); ax.imshow(cm,cmap='Blues',vmin=0,vmax=1361)
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>680 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',
               title=f'{r["model"].upper()} seed {r["seed"]}: TEST n=2141')
    fig.savefig(OUT/'test_confusion_matrices.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,metric in zip(axes,['accuracy','macro_f1','weighted_f1']):
        for i,kind in enumerate(['cnn','gru']):
            values=table.loc[table.model.eq(kind),metric].to_numpy()
            ax.scatter([i-.12,i,i+.12],values,label=kind.upper())
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),color='black',fmt='s',capsize=5)
        ax.set(xticks=[0,1],xticklabels=['CNN','GRU'],ylim=(0,1),title='Test '+metric,ylabel='Score')
    fig.suptitle('Each point is one seed; black marker = mean +/- sample SD (same test tweets)')
    fig.savefig(OUT/'test_model_comparison.png',dpi=160); plt.close(fig)
    val=pd.read_csv(ROOT/'results/stage5/seed_comparison.csv')
    comparison=table.merge(val[['model','seed','accuracy','macro_f1']],on=['model','seed'],suffixes=('_test','_validation'))
    comparison.to_csv(OUT/'validation_test_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,kind in zip(axes,['cnn','gru']):
        f=comparison.loc[comparison.model.eq(kind)]
        ax.plot(f.seed,f.macro_f1_validation,'o-',label='Validation (used for selection)')
        ax.plot(f.seed,f.macro_f1_test,'s-',label='Final test')
        ax.set(xticks=[42,43,44],ylim=(0,1),xlabel='Seed',ylabel='Macro-F1',title=kind.upper());ax.legend(fontsize=8)
    fig.savefig(OUT/'validation_test_comparison.png',dpi=160);plt.close(fig)
