"""Fixed baseline runs: train and validation only; no test loading or tuning."""
from pathlib import Path
import hashlib, json, time, random, platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from models import fresh_model, TrainingWordMasker

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage4'
DATA=ROOT/'data/processed/stage2'
CLASSES=['negative','neutral','positive']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')
def preserved():
    paths=[p for d in ['data/processed/stage2','results/stage2','results/stage3'] for p in (ROOT/d).glob('*') if p.is_file()]
    paths += [ROOT/'notebooks/02_preprocessing_and_split.ipynb',ROOT/'notebooks/03_model_verification.ipynb',ROOT/'src/models.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def load_array(name):
    assert name in ['train','validation']
    with np.load(DATA/f'{name}_encoded.npz',allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def batch(a,ix):
    lengths=torch.tensor(a['lengths'][ix],dtype=torch.long)
    return (torch.tensor(a['input_ids'][ix,:int(lengths.max())],dtype=torch.long),lengths,
            torch.tensor(a['labels'][ix],dtype=torch.long),a['source_rows'][ix].tolist())
def metrics(y,p): return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0))}
def evaluate(model,a,weights):
    model.eval(); logits=[]; numerator=0.; denominator=0.
    with torch.no_grad():
        for start in range(0,len(a['labels']),32):
            x,l,y,_=batch(a,np.arange(start,min(start+32,len(a['labels']))))
            scores=model(x,l)
            numerator+=float(nn.functional.cross_entropy(scores,y,weight=weights,reduction='sum'))
            denominator+=float(weights[y].sum()); logits.append(scores.numpy())
    scores=np.concatenate(logits); p=scores.argmax(1)
    return {'loss':numerator/denominator,**metrics(a['labels'],p)},scores

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    before=preserved(); wall=time.perf_counter()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    assert json.loads((ROOT/'results/stage3/verification.json').read_text())['all_passed']
    train,val=load_array('train'),load_array('validation')
    vocab=json.loads((DATA/'vocabulary.json').read_text(encoding='utf-8'))
    counts=np.bincount(train['labels'],minlength=3)
    weights=torch.tensor(len(train['labels'])/(3*counts),dtype=torch.float32)
    references={str(p.relative_to(ROOT)):sha(p) for p in [DATA/'vocabulary.json',DATA/'config.json',DATA/'split_manifest.csv',ROOT/'src/models.py']}
    settings={'seed':42,'ordering':'numpy default_rng SeedSequence([42, epoch]); shared permutation per epoch',
              'masking':'2% training-only, seed42, epoch indexed from 0, shared TrainingWordMasker',
              'optimizer':'Adam','learning_rate':.001,'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0,
              'batch_size':32,'dropout':.3,'gradient_clip_norm':1.,'max_epochs':20,'patience':3,
              'significant_improvement':.0001,'checkpoint_rule':'highest validation macro-F1, earliest exact tie',
              'patience_rule':'reset when macro-F1 exceeds significant-improvement anchor by >0.0001',
              'class_counts_train':counts.tolist(),'class_weights_train':weights.tolist(),
              'loss':'weighted CE; epoch loss=sum weighted per-example losses / sum observed target weights',
              'threads':2,'workers':0,'dtype':'float32','device':'cpu','length_cap':46,
              'python':platform.python_version(),'torch':str(torch.__version__),'references':references}
    save('configuration.json',settings)
    masker=TrainingWordMasker(vocab,seed=42)
    outcomes={}; histories={}; initial_embeddings=[]
    for kind in ['cnn','gru']:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model=fresh_model(kind,len(vocab),42)
        initial_embeddings.append(sha_bytes(model.embedding.weight.detach().numpy().tobytes()))
        initial_unk=model.embedding.weight[1].detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        history=[]; best=-1.; anchor=-1.; stale=0; best_epoch=None; best_scores=None
        model_start=time.perf_counter()
        for epoch in range(20):
            epoch_start=time.perf_counter(); model.train()
            order=np.random.default_rng(np.random.SeedSequence([42,epoch])).permutation(len(train['labels']))
            truth=[]; predictions=[]; numerator=0.; denominator=0.; masked_count=0
            for start in range(0,len(order),32):
                ix=order[start:start+32]; x,l,y,rows=batch(train,ix)
                x,changed=masker(x,rows,epoch,split='train',training=True); masked_count+=int(changed.sum())
                optimizer.zero_grad(set_to_none=True); scores=model(x,l)
                loss=nn.functional.cross_entropy(scores,y,weight=weights)
                assert torch.isfinite(loss)
                loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert torch.isfinite(norm)
                optimizer.step()
                numerator+=float(loss.detach())*float(weights[y].sum()); denominator+=float(weights[y].sum())
                truth.extend(y.tolist()); predictions.extend(scores.detach().argmax(1).tolist())
            assert not model.embedding.weight[0].any()
            train_metrics={'loss':numerator/denominator,**metrics(truth,predictions)}
            validation,scores=evaluate(model,val,weights)
            row={'epoch':epoch+1,**{f'train_masked_{k}':v for k,v in train_metrics.items()},
                 **{f'validation_unmasked_{k}':v for k,v in validation.items()},'masked_tokens':masked_count,
                 'order_sha256':sha_bytes(order.tobytes())}
            score=validation['macro_f1']
            if score>best:
                best=score; best_epoch=epoch+1; best_scores=scores.copy()
                torch.save({'state_dict':model.state_dict(),'kind':kind,'epoch':best_epoch,'validation_macro_f1':best,
                            'settings':settings,'vocabulary_size':len(vocab),'references':references},OUT/f'{kind}_best.pt')
                np.savez_compressed(OUT/f'{kind}_best_validation.npz',logits=best_scores,predictions=best_scores.argmax(1),
                                    labels=val['labels'],source_rows=val['source_rows'])
            if score>anchor+.0001: anchor=score; stale=0
            else: stale+=1
            row.update({'elapsed_seconds':time.perf_counter()-epoch_start,'patience_counter':stale,'best_epoch_so_far':best_epoch})
            history.append(row); pd.DataFrame(history).to_csv(OUT/f'{kind}_history.csv',index=False)
            print(f'{kind} epoch {epoch+1}: masked train F1={train_metrics["macro_f1"]:.4f}, validation F1={score:.4f}, stale={stale}, seconds={row["elapsed_seconds"]:.2f}',flush=True)
            if stale>=3: break
        training_seconds=time.perf_counter()-model_start
        assert not torch.equal(initial_unk,model.embedding.weight[1])
        del model,optimizer
        # New instance, checkpoint loading, independent full-validation forward pass.
        checkpoint=torch.load(OUT/f'{kind}_best.pt',map_location='cpu',weights_only=True)
        restored=fresh_model(kind,len(vocab),seed=999)
        restored.load_state_dict(checkpoint['state_dict'])
        reloaded_metrics,reloaded_scores=evaluate(restored,val,weights)
        saved=np.load(OUT/f'{kind}_best_validation.npz',allow_pickle=False)
        assert np.array_equal(saved['predictions'],reloaded_scores.argmax(1))
        assert np.array_equal(saved['logits'],reloaded_scores)
        report=classification_report(val['labels'],reloaded_scores.argmax(1),labels=[0,1,2],target_names=CLASSES,output_dict=True,zero_division=0)
        matrix=confusion_matrix(val['labels'],reloaded_scores.argmax(1),labels=[0,1,2]).tolist()
        save(f'{kind}_validation_report.json',{'classification_report':report,'confusion_matrix':matrix,'class_order':CLASSES})
        outcomes[kind]={'best_epoch':best_epoch,'epochs_run':len(history),'validation':reloaded_metrics,
                        'training_and_validation_seconds':training_seconds,'parameters':sum(p.numel() for p in restored.parameters()),
                        'checkpoint_sha256':sha(OUT/f'{kind}_best.pt'),'reload_predictions_exact':True,'reload_logits_exact':True,
                        'pad_zero':bool(not restored.embedding.weight[0].any()),'unk_updated_during_training':True,
                        'per_class':{c:report[c] for c in CLASSES},'confusion_matrix':matrix}
        histories[kind]=history; del restored,checkpoint
    common=min(len(histories['cnn']),len(histories['gru']))
    fair=all(histories['cnn'][e]['order_sha256']==histories['gru'][e]['order_sha256'] and
             histories['cnn'][e]['masked_tokens']==histories['gru'][e]['masked_tokens'] for e in range(common))
    assert fair and initial_embeddings[0]==initial_embeddings[1]
    assert before==preserved()
    save('preservation.json',{'before':before,'after':preserved(),'identical':True})
    summary={'models':outcomes,'shared_order_and_mask_counts_verified':fair,'same_initial_embedding':True,
             'previous_outputs_preserved':True,'test_evaluated':False,'tuning_started':False,
             'total_run_seconds':time.perf_counter()-wall,'total_model_training_validation_seconds':sum(r['training_and_validation_seconds'] for r in outcomes.values())}
    save('summary.json',summary)
    pd.DataFrame([{'model':k,'best_epoch':v['best_epoch'],'epochs_run':v['epochs_run'],**v['validation'],
                   'training_validation_seconds':v['training_and_validation_seconds']} for k,v in outcomes.items()]).to_csv(OUT/'baseline_comparison.csv',index=False)
    plots(histories,outcomes)
    return summary

def sha_bytes(value): return hashlib.sha256(value).hexdigest()

def plots(histories,outcomes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,kind in enumerate(['cnn','gru']):
        frame=pd.DataFrame(histories[kind])
        for j,metric in enumerate(['loss','accuracy','macro_f1']):
            ax=axes[i,j]
            ax.plot(frame.epoch,frame[f'train_masked_{metric}'],label='Train: masked, dropout on')
            ax.plot(frame.epoch,frame[f'validation_unmasked_{metric}'],label='Validation: unmasked, eval')
            ax.axvline(outcomes[kind]['best_epoch'],color='grey',linestyle=':',label='Best validation F1 epoch')
            ax.set(title=f'{kind.upper()} {metric}',xlabel='Epoch'); ax.grid(alpha=.2)
            if j: ax.set_ylim(0,1)
            if i==0 and j==0: ax.legend(fontsize=7)
    fig.savefig(OUT/'learning_curves.png',dpi=160); fig.savefig(OUT/'learning_curves.svg'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,kind in zip(axes,['cnn','gru']):
        matrix=np.array(outcomes[kind]['confusion_matrix']); ax.imshow(matrix,cmap='Blues')
        for i in range(3):
            for j in range(3): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title=f'{kind.upper()} best-checkpoint validation')
    fig.savefig(OUT/'validation_confusion_matrices.png',dpi=160); plt.close(fig)
