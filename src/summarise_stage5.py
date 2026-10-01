"""Write working findings from completed saved results; no model computation."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage5'
s=json.loads((OUT/'summary.json').read_text())
t=pd.read_csv(OUT/'tuning_comparison.csv')
lines=['# Stage 5 tuning findings','',
       'The fixed grid and additional-seed runs are complete. Eight grid entries comprise two exactly matched reused baselines and six fresh runs; four more fresh runs assess the selected configurations at seeds 43 and 44. All trial checkpoints reload to exactly reproduce their saved validation logits and predictions. Earlier stages and the fixed split/preprocessing remain unchanged. No test performance was evaluated.','',
       '## Seed-42 grid results','',
       '| Model | Learning rate | Dropout | Best epoch | Epochs run | Validation accuracy | Validation macro-F1 | New training/validation seconds |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in t.itertuples():
    lines.append(f'| {r.model.upper()}'+(' (baseline reused)' if r.reused else '')+f' | {r.lr:g} | {r.dropout:.1f} | {r.best_epoch} | {r.epochs_run} | {r.accuracy:.4f} | {r.macro_f1:.6f} | {r.new_training_seconds:.2f} |')
lines += ['','Selection was fixed before training: highest seed42 validation macro-F1, earlier best epoch on an exact tie, then listed grid order. No tie tolerance or other metric was substituted. The additional seeds did not change configuration choices.','']
for kind,c in s['selected'].items():
    base=next(r for r in s['grid_trials'] if r['kind']==kind and r['reused_baseline'])
    delta=c['validation']['macro_f1']-base['validation']['macro_f1']
    lines.append(f"{kind.upper()}: selected learning rate {c['learning_rate']:g}, dropout {c['dropout']:.1f}. Validation macro-F1 change from baseline: {delta:+.6f} ({delta*100:+.3f} percentage points). "+
                 ('The baseline remains selected; the search did not improve it.' if c['reused_baseline'] else 'This is a validation-selected improvement, not a demonstrated test gain.'))
lines += ['','## Selected configurations across three seeds','',
          '| Model | Seed 42 F1 | Seed 43 F1 | Seed 44 F1 | Mean F1 | Sample SD |','|---|---:|---:|---:|---:|---:|']
for kind,r in s['variability'].items():
    a,b,c=r['validation_macro_f1']
    lines.append(f"| {kind.upper()} | {a:.6f} | {b:.6f} | {c:.6f} | {r['mean']:.6f} | {r['sample_std']:.6f} |")
lines += ['', 'All three seeds are included. Standard deviation uses ddof=1. These runs change random initialisation, dropout, ordering and mask draws while retaining the exact split and selected hyperparameters. They assess training randomness on one dataset partition. They are not three independent datasets, and three runs do not establish significance. Seed42 helped select the configuration, making the combined validation mean subject to selection bias. No seed is designated the final winner.','',
          '## Runtime and interpretation','',
          f"The ten fresh runs used {s['new_training_validation_seconds']:.2f} seconds ({s['new_training_validation_seconds']/60:.2f} minutes) of training and validation. Stage 5 wall time through checkpoint selection/freezing, before plot generation, was {s['stage5_wall_seconds_before_plots']:.2f} seconds. Reused baseline training cost zero new time; its historical runtime is retained separately in the comparison CSV. Later independent verification and notebook/figure saving are outside these counters.", '',
          'All configurations used comparable trial budgets and the same protocol: CPU float32, two threads, batch32, Adam, training-weighted CE, clipping1.0, cap20 epochs and patience3. Learning rate and dropout were the only grid changes. Actual epochs differ due to early stopping; equal trial/epoch caps do not imply equal CPU time.', '',
          'This small grid cannot show that these are globally optimal settings. Stronger dropout and a lower learning rate were motivated by baseline overfitting, but may slow learning under the fixed stopping rule. Negative results are retained. Do not extend only unsuccessful trials or add new settings after inspecting these results. Masked training metrics and unmasked validation metrics have different conditions; logs use explicit names to avoid confusing them. Near-duplicates, annotation quality and the known unseen-word encoded collision remain limitations.', '',
          '![Grid results](results/stage5/grid_comparison.png)','',
          '![Seed variability](results/stage5/seed_variability.png)','',
          '## Frozen final-test proposal','',
          'The exact six future test checkpoints and hashes are recorded in results/stage5/frozen_test_protocol.json: selected CNN and GRU configurations, each at seeds42/43/44, using their best-validation epochs. Report all six results and per-model means/sample SD, never the luckiest seed. Do not refit on train+validation or substitute checkpoints. Test inference will use eval mode, no masking/gradients, fixed vocabulary/cap, and argmax logits. Report seed-wise accuracy, macro-F1, per-class precision/recall/F1 and confusion matrices; make paired-seed model comparisons descriptive. No test evaluation has run.','',
          'Read [full protocol and rerun instructions](docs/stage5_protocol.md) before final evaluation. It fixes the reporting policy and forbids test-driven changes. Future test access still requires the next authorised stage.','',
          '## Saved work and understanding','',
          'Executed notebook: notebooks/05_tuning_and_seed_variability.ipynb. Source: src/stage5.py and src/run_stage5_notebook.py, using unchanged src/models.py and Stage 4 batch/validation helpers. Trial folders under results/stage5/ contain every configuration, epoch log, best checkpoint, validation prediction array, class report and actual runtime. Root tables/plots compare the grid and seed runs. src/verify_stage5.py independently checks saved settings, early stopping, selection, checkpoint predictions and sample-SD arithmetic without more training.','',
          'The primary marking scheme and brief support justified tuning; filename/section and transcript page references are in docs/stage5_protocol.md. This working record is not the assignment report. Codex assistance includes code generation/modification and execution of tuning and repeat training; no test evaluation or AI acknowledgement was written.','',
          'Understand that a tuning search may produce no improvement, validation selection can be optimistic, seed variability is not a confidence interval over future tweets, and the final test must assess the already frozen choices.']
(ROOT/'STAGE5.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('STAGE5.md written from saved completed outputs.')
