"""Training-only diagnostics, not baseline training or tuning. Never save weights."""
from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import platform
import time
import numpy as np
import torch
from torch import nn
from models import fresh_model, TrainingWordMasker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/stage3'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def stage2_hashes():
    paths = [p for folder in ['data/processed/stage2','results/stage2'] for p in (ROOT/folder).glob('*') if p.is_file()]
    paths += [ROOT/'notebooks/02_preprocessing_and_split.ipynb', ROOT/'config/stage2_frozen_inputs.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2), encoding='utf-8')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    before = stage2_hashes()
    verified = json.loads((ROOT/'results/stage2/independent_verification.json').read_text())
    assert verified['structural_verification_passed']
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    vocab = json.loads((ROOT/'data/processed/stage2/vocabulary.json').read_text(encoding='utf-8'))
    config = json.loads((ROOT/'data/processed/stage2/config.json').read_text())
    # Only TRAIN arrays are loaded for model computation. Held-out files are hashed only.
    with np.load(ROOT/'data/processed/stage2/train_encoded.npz', allow_pickle=False) as z:
        arrays = {k:z[k].copy() for k in z.files}
    assert len(arrays['labels']) == 9989 and config['length_cap'] == 46
    masker = TrainingWordMasker(vocab)
    weights = torch.tensor([config['optional_training_class_weights'][s] for s in ['negative','neutral','positive']])
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        assert value, name
    def batch(indices):
        lengths = torch.tensor(arrays['lengths'][indices], dtype=torch.long)
        ids = torch.tensor(arrays['input_ids'][indices, :int(lengths.max())], dtype=torch.long)
        return ids, lengths, torch.tensor(arrays['labels'][indices], dtype=torch.long), arrays['source_rows'][indices].tolist()

    # Evaluate the actual 2% policy across the training partition, without fitting a model.
    ids, lengths, labels, rows = batch(np.arange(len(arrays['labels'])))
    masked, positions = masker(ids, rows, 0, split='train', training=True)
    eligible_count = int(masker.eligible[ids].sum())
    check('mask_input_unchanged', np.array_equal(ids.numpy(), arrays['input_ids']))
    check('protected_tokens_unchanged', torch.equal(masked[~masker.eligible[ids]], ids[~masker.eligible[ids]]))
    check('mask_exposure_positive', positions.any())
    check('mask_reproducible', torch.equal(masked, masker(ids, rows, 0, split='train', training=True)[0]))
    rev = list(range(31,-1,-1))
    check('mask_batch_order_independent', torch.equal(masked[:32][rev], masker(ids[:32][rev], [rows[i] for i in rev], 0, split='train', training=True)[0]))
    for split, training in [('validation',True),('test',True),('train',False)]:
        # Synthetic split flags on TRAIN inputs; no held-out examples are used.
        out, changed = masker(ids[:32], rows[:32], 0, split=split, training=training)
        check(f'masking_disabled_{split}_{training}', torch.equal(out,ids[:32]) and not changed.any())
    # Force probability=1 in a synthetic unit test to prove eligibility protections.
    tokens = ['<PAD>','<UNK>','<URL>','<USER>','no','NOT','never',"can't",'don\u2019t','!', '?','\U0001f621','#happy','ordinary','123']
    synthetic_vocab={t:i for i,t in enumerate(tokens)}
    probe=torch.arange(len(tokens))[None,:]
    probe_out,_=TrainingWordMasker(synthetic_vocab,probability=1)(probe,['synthetic'],0,split='train',training=True)
    check('explicit_protected_negation_punctuation_emoji', torch.equal(probe_out[0,:12],probe[0,:12]) and torch.all(probe_out[0,12:]==1))
    mask_stats={'probability':0.02,'eligible_training_positions':eligible_count,'masked_positions_epoch0':int(positions.sum()),
                'observed_fraction':int(positions.sum())/eligible_count,'protected_policy':'reserved IDs, negation/nt contractions, punctuation, emojis',
                'seed':42,'scope':'training only; same stateless policy for both models'}
    del masked, ids, lengths, labels

    # Stratified tiny subset, selected only from training: 8 examples per class.
    tiny_idx=np.concatenate([np.flatnonzero(arrays['labels']==c)[:8] for c in range(3)])
    tiny_ids,tiny_lengths,tiny_y,tiny_rows=batch(tiny_idx)
    save('diagnostic_source_rows.json',{'overfit':tiny_rows,'runtime':arrays['source_rows'][np.random.default_rng(42).permutation(9989)[:800]].tolist()})
    results={}
    embedding_hashes=[]
    for kind, expected_count in [('cnn',902019),('gru',914627)]:
        torch.manual_seed(42)
        model=fresh_model(kind,len(vocab))
        embedding_hashes.append(hashlib.sha256(model.embedding.weight.detach().numpy().tobytes()).hexdigest())
        count=sum(p.numel() for p in model.parameters())
        check(kind+'_parameter_count',count==expected_count)
        model.eval()
        short=torch.tensor([[4,0,0,0],[5,6,0,0],[7,8,9,10]])
        lens=torch.tensor([1,2,4])
        with torch.no_grad():
            scores=model(short,lens)
            extra=model(torch.nn.functional.pad(short,(0,7)),lens)
            separate=torch.cat([model(short[i:i+1,:int(l)],lens[i:i+1]) for i,l in enumerate(lens)])
        padding_difference=float((scores-extra).abs().max())
        check(kind+'_short_mixed_shape_finite',scores.shape==(3,3) and torch.isfinite(scores).all())
        check(kind+'_extra_padding_invariant',torch.allclose(scores,extra,atol=1e-6,rtol=1e-5))
        check(kind+'_batch_composition_invariant',torch.allclose(scores,separate,atol=1e-6,rtol=1e-5))
        try:
            model(torch.zeros((1,2),dtype=torch.long),torch.tensor([0]))
            rejected=False
        except ValueError: rejected=True
        check(kind+'_empty_input_rejected',rejected)
        # Real training subset with actual deterministic 2% UNK exposure.
        exposure_idx=np.flatnonzero(positions.any(dim=1).numpy())[:32]
        x,l,y,r=batch(exposure_idx)
        x,changed=masker(x,r,0,split='train',training=True)
        model.train(); optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        unk_before=model.embedding.weight[1].detach().clone()
        head_before=model.head.weight.detach().clone()
        loss=nn.functional.cross_entropy(model(x,l),y,weight=weights)
        loss.backward()
        unk_gradient=float(model.embedding.weight.grad[1].norm())
        check(kind+'_finite_loss_gradients',torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        check(kind+'_unk_nonzero_gradient',unk_gradient>0)
        check(kind+'_pad_zero_gradient',not model.embedding.weight.grad[0].any())
        optimizer.step()
        unk_update=float((model.embedding.weight[1].detach()-unk_before).norm())
        check(kind+'_unk_updated',unk_update>0)
        check(kind+'_head_updated',not torch.equal(head_before,model.head.weight))
        check(kind+'_pad_stays_zero',not model.embedding.weight[0].any())
        del model, optimizer

        # Fresh diagnostic model. Dropout/masking disabled ONLY for this memorisation test.
        # Backprop works in eval mode. Plain CE avoids class weights on the balanced subset.
        model=fresh_model(kind,len(vocab)); model.eval()
        optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
        with torch.no_grad(): initial_loss=float(nn.functional.cross_entropy(model(tiny_ids,tiny_lengths),tiny_y))
        curve=[]
        for step in range(1,101):
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(tiny_ids,tiny_lengths),tiny_y)
            loss.backward(); optimizer.step()
            if step%5==0:
                with torch.no_grad():
                    output=model(tiny_ids,tiny_lengths)
                    current_loss=float(nn.functional.cross_entropy(output,tiny_y))
                    accuracy=float((output.argmax(1)==tiny_y).float().mean())
                curve.append({'step':step,'loss':current_loss,'accuracy':accuracy})
                if accuracy==1.0 and current_loss<0.05: break
        check(kind+'_tiny_subset_learns',accuracy>=0.95 and current_loss<initial_loss*0.2)
        check(kind+'_pad_zero_after_overfit',not model.embedding.weight[0].any())
        overfit={'rows':24,'class_counts':[8,8,8],'maximum_steps':100,'steps':step,'initial_loss':initial_loss,
                 'final_loss':current_loss,'training_subset_accuracy':accuracy,'learning_rate':0.01,
                 'dropout_and_masking':False,'diagnostic_only':True,'curve':curve}
        del model,optimizer

        # Fresh model, real 2% masking/dropout and weighted CE; timing includes batch
        # creation, masking, forward, backward, clipping and Adam update, not import/init.
        model=fresh_model(kind,len(vocab)); model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
        order=np.random.default_rng(42).permutation(9989)
        times=[]; masked_count=0
        for b in range(25):
            start=time.perf_counter()
            x,l,y,r=batch(order[b*32:(b+1)*32])
            x,changed=masker(x,r,0,split='train',training=True)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(x,l),y,weight=weights)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            elapsed=time.perf_counter()-start
            if b>=5: times.append(elapsed)
            masked_count+=int(changed.sum())
        check(kind+'_runtime_loss_finite',torch.isfinite(loss))
        check(kind+'_pad_zero_after_runtime',not model.embedding.weight[0].any())
        timing={'warmup_batches':5,'timed_batches':20,'batch_size':32,'threads':torch.get_num_threads(),
                'loader_workers':0,'dtype':'float32','batch_seconds':times,'mean_batch_seconds':float(np.mean(times)),
                'median_batch_seconds':float(np.median(times)),'minimum_batch_seconds':min(times),'maximum_batch_seconds':max(times),
                'estimated_313_batch_training_epoch_seconds':float(np.mean(times)*math.ceil(9989/32)),
                'estimate_excludes':'validation, checkpoints, startup; short sample subject to CPU load and length variation',
                'masked_positions_25_batches':masked_count}
        results[kind]={'parameters':count,'fixed_pad_entries':64,'parameter_bytes_float32':count*4,
                       'padding_max_abs_difference':padding_difference,'unk_gradient_norm':unk_gradient,'unk_update_norm':unk_update,
                       'overfit_diagnostic':overfit,'cpu_timing':timing}
        del model,optimizer
        gc.collect()
        print(kind, 'verified; diagnostic accuracy',accuracy,'mean batch seconds',timing['mean_batch_seconds'],flush=True)
    check('same_initial_embedding_for_both_models',embedding_hashes[0]==embedding_hashes[1])
    check('same_mask_exposure_both_runtime_checks',results['cnn']['cpu_timing']['masked_positions_25_batches']==results['gru']['cpu_timing']['masked_positions_25_batches'])
    check('stage2_unchanged',before==stage2_hashes())
    save('stage2_preservation.json',{'before':before,'after':stage2_hashes(),'identical':checks['stage2_unchanged']})
    report={'executed_utc':datetime.now(timezone.utc).isoformat(),'environment':{'python':platform.python_version(),
             'torch':torch.__version__,'platform':platform.platform(),'processor':platform.processor(),'device':'cpu','threads':torch.get_num_threads()},
            'masking':mask_stats,'models':results,'checks':checks,'all_passed':all(checks.values()),
            'weights_saved':False,'diagnostic_weights_discarded':True,'baseline_training_started':False,
            'test_performance_accessed':False,'scope':'training-only diagnostic learning and runtime; not baseline evaluation'}
    save('verification.json',report)
    return report
