"""Prespecified Stage 5 grid and seed repeats. Train/validation only."""
from pathlib import Path
import copy, hashlib, json, time, random, platform, shutil
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
from models import fresh_model, TrainingWordMasker
from stage4 import ROOT, DATA, CLASSES, sha, sha_bytes, batch, metrics, evaluate, load_array, preserved

OUTER=ROOT/'results/stage5'
BASE=ROOT/'results/stage4'
GRID=[(.001,.3),(.001,.5),(.0005,.3),(.0005,.5)]
SEEDS=[42,43,44]
def write(path,obj): path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def earlier_hashes():
    return {**preserved(),**{str(p.relative_to(ROOT)):sha(p) for p in BASE.rglob('*') if p.is_file()},
            **{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/stage4.py',ROOT/'notebooks/04_baseline_training.ipynb']}}

def train_trial(kind,lr,dropout,seed,OUT,train,val,vocab,weights,settings):
    OUT.mkdir(parents=True,exist_ok=True)
    save=lambda name,obj:write(OUT/name,obj)
    references=settings['references']
    save('configuration.json',settings)
    masker=TrainingWordMasker(vocab,seed=seed)
    outcomes={}; histories={}; initial_embeddings=[]
    for kind in [kind]:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model=fresh_model(kind,len(vocab),seed)
        model.dropout.p=dropout
        initial_embeddings.append(sha_bytes(model.embedding.weight.detach().numpy().tobytes()))
        initial_unk=model.embedding.weight[1].detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0)
        history=[]; best=-1.; anchor=-1.; stale=0; best_epoch=None; best_scores=None
        model_start=time.perf_counter()
        for epoch in range(20):
            epoch_start=time.perf_counter(); model.train()
            order=np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(len(train['labels']))
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
        restored.dropout.p=dropout
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
    result=outcomes[kind]
    result.update({'kind':kind,'learning_rate':lr,'dropout':dropout,'seed':seed,'reused_baseline':False,
                   'initial_embedding_sha256':initial_embeddings[0],'directory':str(OUT.relative_to(ROOT))})
    save('result.json',result)
    return result

def run():
    OUTER.mkdir(parents=True,exist_ok=True)
    before=earlier_hashes(); started=time.perf_counter()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    train,val=load_array('train'),load_array('validation')
    vocab=json.loads((DATA/'vocabulary.json').read_text(encoding='utf-8'))
    counts=np.bincount(train['labels'],minlength=3)
    weights=torch.tensor(len(train['labels'])/(3*counts),dtype=torch.float32)
    base_settings=json.loads((BASE/'configuration.json').read_text())
    expected={'seed':42,'optimizer':'Adam','learning_rate':.001,'dropout':.3,'batch_size':32,
              'gradient_clip_norm':1.,'max_epochs':20,'patience':3,'significant_improvement':.0001,
              'betas':[.9,.999],'epsilon':1e-8,'weight_decay':0,'threads':2,'workers':0,'dtype':'float32',
              'device':'cpu','length_cap':46,'class_counts_train':counts.tolist(),'class_weights_train':weights.tolist(),
              'python':platform.python_version(),'torch':str(torch.__version__)}
    assert all(base_settings[k]==v for k,v in expected.items())
    record=json.loads((BASE/'notebook_execution.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in record['source_hashes'].items())
    assert all(sha(ROOT/p)==h for p,h in base_settings['references'].items())
    assert base_settings['ordering']=='numpy default_rng SeedSequence([42, epoch]); shared permutation per epoch'
    assert base_settings['masking']=='2% training-only, seed42, epoch indexed from 0, shared TrainingWordMasker'
    assert base_settings['checkpoint_rule']=='highest validation macro-F1, earliest exact tie'
    assert base_settings['patience_rule']=='reset when macro-F1 exceeds significant-improvement anchor by >0.0001'
    assert base_settings['loss']=='weighted CE; epoch loss=sum weighted per-example losses / sum observed target weights'
    plan={'grid':[{'learning_rate':lr,'dropout':drop} for lr,drop in GRID],'grid_seed':42,'repeat_seeds':[43,44],
          'selection':'highest seed42 validation macro-F1; exact tie: earlier best epoch, then listed grid order',
          'repeats':'fixed selected hyperparameters, no reselection and no luckiest seed; report all seeds and sample SD ddof=1',
          'fixed_baseline_configuration':base_settings,'baseline_protocol_reuse_verified':True,
          'new_grid_runs':6,'additional_seed_runs':4,'test_evaluation':False}
    write(OUTER/'prespecified_plan.json',plan)  # Written before any new training.
    baseline=json.loads((BASE/'summary.json').read_text())
    trials=[]; selected={}; repeats={}
    def config(lr,drop,seed):
        c=copy.deepcopy(base_settings); c.update(seed=seed,learning_rate=lr,dropout=drop)
        c['ordering']=f'numpy default_rng SeedSequence([{seed}, epoch]); shared permutation per epoch'
        c['masking']=f'2% training-only, seed{seed}, epoch indexed from 0, shared TrainingWordMasker'
        return c
    for kind in ['cnn','gru']:
        for i,(lr,drop) in enumerate(GRID):
            folder=OUTER/f'{kind}_grid{i}_seed42'
            if i==0:
                ckpt=torch.load(BASE/f'{kind}_best.pt',map_location='cpu',weights_only=True)
                assert ckpt['settings']==base_settings and ckpt['kind']==kind
                assert sha(BASE/f'{kind}_best.pt')==baseline['models'][kind]['checkpoint_sha256']
                folder.mkdir(exist_ok=True)
                for suffix in ['best.pt','best_validation.npz','history.csv','validation_report.json']:
                    shutil.copy2(BASE/f'{kind}_{suffix}',folder/f'{kind}_{suffix}')
                write(folder/'configuration.json',base_settings)
                model=fresh_model(kind,len(vocab)); model.load_state_dict(ckpt['state_dict'])
                m,scores=evaluate(model,val,weights)
                with np.load(folder/f'{kind}_best_validation.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['logits'],scores) and np.array_equal(z['predictions'],scores.argmax(1))
                assert m==baseline['models'][kind]['validation']
                result=copy.deepcopy(baseline['models'][kind])
                result.update(kind=kind,learning_rate=lr,dropout=drop,seed=42,reused_baseline=True,
                              directory=str(folder.relative_to(ROOT)),stage5_training_seconds=0.)
                write(folder/'result.json',result)
                del model,ckpt
            else:
                result=train_trial(kind,lr,drop,42,folder,train,val,vocab,weights,config(lr,drop,42))
            result['grid_index']=i; trials.append(result)
            write(OUTER/'grid_progress.json',trials)
        candidates=[r for r in trials if r['kind']==kind]
        selected[kind]=min(candidates,key=lambda r:(-r['validation']['macro_f1'],r['best_epoch'],r['grid_index']))
    write(OUTER/'selected_before_repeats.json',selected)
    for kind in ['cnn','gru']:
        choice=selected[kind]; repeats[kind]=[choice]
        for seed in [43,44]:
            folder=OUTER/f'{kind}_selected_seed{seed}'
            repeats[kind].append(train_trial(kind,choice['learning_rate'],choice['dropout'],seed,folder,
                                            train,val,vocab,weights,config(choice['learning_rate'],choice['dropout'],seed)))
            write(OUTER/'repeat_progress.json',repeats)
    all_runs=trials+[r for runs in repeats.values() for r in runs if r['seed']!=42]
    # Same epoch permutations and masking counts for all trials sharing each seed.
    for seed in SEEDS:
        histories=[pd.read_csv(ROOT/r['directory']/f'{r["kind"]}_history.csv') for r in all_runs if r['seed']==seed]
        for i,h in enumerate(histories):
            for other in histories[i+1:]:
                n=min(len(h),len(other))
                assert h.order_sha256.iloc[:n].tolist()==other.order_sha256.iloc[:n].tolist()
                assert h.masked_tokens.iloc[:n].tolist()==other.masked_tokens.iloc[:n].tolist()
    variability={kind:{'seeds':SEEDS,'validation_macro_f1':[r['validation']['macro_f1'] for r in runs],
                       'mean':float(np.mean([r['validation']['macro_f1'] for r in runs])),
                       'sample_std':float(np.std([r['validation']['macro_f1'] for r in runs],ddof=1))} for kind,runs in repeats.items()}
    frozen=[]
    for kind,runs in repeats.items():
        for r in runs:
            p=ROOT/r['directory']/f'{kind}_best.pt'
            frozen.append({'model':kind,'seed':r['seed'],'learning_rate':r['learning_rate'],'dropout':r['dropout'],
                           'best_validation_epoch':r['best_epoch'],'checkpoint':str(p.relative_to(ROOT)),'sha256':sha(p)})
    write(OUTER/'frozen_test_protocol.json',{
        'status':'choices frozen before any test performance access; test evaluation pending authorization',
        'checkpoints':frozen,'data_references':base_settings['references'],
        'test_file_reference':{'path':str((DATA/'test_encoded.npz').relative_to(ROOT)),'sha256':sha(DATA/'test_encoded.npz')},
        'evaluation':'all six checkpoints, eval mode, no masking/no gradients, fixed vocabulary/cap, argmax logits, batch32',
        'report':'each seed accuracy/macro-F1, per-class precision/recall/F1 and confusion matrix; per-model mean and sample SD across all three seeds',
        'comparison':'paired seeds for model differences; report all seeds; no best-seed selection or ensemble',
        'after_test':'no model, checkpoint, threshold, preprocessing or reporting-seed changes based on test results',
        'no_refit':'do not retrain on train+validation; evaluate exactly the frozen checkpoints'})
    summary={'selected':selected,'variability':variability,'grid_trials':trials,'seed_runs':repeats,
             'new_training_validation_seconds':sum(r['training_and_validation_seconds'] for r in all_runs if not r['reused_baseline']),
             'stage5_wall_seconds_before_plots':time.perf_counter()-started,'baseline_training_not_repeated':True,
             'same_order_and_masking_verified':True,'all_reload_checks_exact':True,'test_evaluated':False,
             'previous_outputs_preserved':before==earlier_hashes()}
    assert summary['previous_outputs_preserved']
    write(OUTER/'preservation.json',{'before':before,'after':earlier_hashes(),'identical':True})
    write(OUTER/'summary.json',summary)
    table=pd.DataFrame([{'model':r['kind'],'grid_index':r['grid_index'],'lr':r['learning_rate'],'dropout':r['dropout'],
                          'seed':r['seed'],'reused':r['reused_baseline'],'best_epoch':r['best_epoch'],'epochs_run':r['epochs_run'],
                          **r['validation'],'historical_training_seconds':r['training_and_validation_seconds'],
                          'new_training_seconds':0 if r['reused_baseline'] else r['training_and_validation_seconds']} for r in trials])
    table.to_csv(OUTER/'tuning_comparison.csv',index=False)
    pd.DataFrame([{'model':kind,'seed':r['seed'],'lr':r['learning_rate'],'dropout':r['dropout'],
                  'best_epoch':r['best_epoch'],**r['validation']} for kind,runs in repeats.items() for r in runs]).to_csv(OUTER/'seed_comparison.csv',index=False)
    make_plots(table,variability)
    return summary

def make_plots(table,variability):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,kind in zip(axes,['cnn','gru']):
        rows=table.loc[table.model.eq(kind)]
        labels=[f'lr={r.lr}\ndropout={r.dropout}'+('\nBaseline reused' if r.reused else '') for r in rows.itertuples()]
        ax.bar(range(4),rows.macro_f1,color=['grey','steelblue','steelblue','steelblue'])
        for i,v in enumerate(rows.macro_f1):ax.text(i,v+.008,f'{v:.4f}',ha='center')
        ax.set(xticks=range(4),xticklabels=labels,ylim=(0,1),ylabel='Validation macro-F1',title=kind.upper()+' seed42 grid')
    fig.savefig(OUTER/'grid_comparison.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for i,(kind,r) in enumerate(variability.items()):
        ax.scatter([i-.12,i,i+.12],r['validation_macro_f1'],label=f'{kind.upper()} seeds42,43,44')
        ax.errorbar(i,r['mean'],yerr=r['sample_std'],color='black',fmt='s',capsize=5)
    ax.set(xticks=[0,1],xticklabels=['CNN selected','GRU selected'],ylabel='Validation macro-F1',
           title='Three seeds: points; mean +/- sample SD',ylim=(0,1)); ax.legend()
    fig.savefig(OUTER/'seed_variability.png',dpi=160); plt.close(fig)
