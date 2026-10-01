"""Create/execute Stage 3 diagnostics in the project venv, preserving prior runs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT=Path(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
out=ROOT/'results/stage3'; out.mkdir(parents=True,exist_ok=True)
path=ROOT/'notebooks/03_model_verification.ipynb'
previous=[p for p in out.glob('*') if p.is_file()]+([path] if path.exists() else [])
snapshot=None
if previous:
    snapshot=ROOT/'reviews'/('stage3_before_rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for p in previous:
        target=snapshot/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
runtime=ROOT/'.jupyter-runtime/stage3'; runtime.mkdir(parents=True,exist_ok=True)
os.environ.update(JUPYTER_RUNTIME_DIR=str(runtime),IPYTHONDIR=str(runtime/'ipython'))
md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md('''# Stage 3: custom PyTorch model verification

Training-only correctness, tiny-subset learning and CPU runtime diagnostics. No baseline training, tuning or test scoring. Stage 2 remains read-only. The runner preserves previous Stage 3 outputs before reruns. Reusable models and masking are in `src/models.py`; diagnostic procedures are in `src/stage3.py`.

Both architectures use random initialisation, a 13,898 x 64 embedding and a three-logit head. CNN: width-3 convolution with 64 filters, ReLU, masked max pooling, dropout 0.3. GRU: one 64-unit recurrent layer with packed lengths, final hidden state and dropout 0.3. PAD is fixed zero. Word masking is 2% on eligible training words only, reproducible by seed/epoch/source row/position, protecting negation, punctuation, emojis and reserved symbols.

The primary marking scheme assigns 20/20/30/10/10/10 marks; see `docs/stage3_proposal.md` for brief sections and Lecture 6/13 transcript page references and caveats. This notebook covers implementation diagnostics, not the full building/tuning/evaluation requirements.'''),
code('''from pathlib import Path
import sys, json
ROOT=Path.cwd().resolve()
if not (ROOT/'src/models.py').exists(): ROOT=ROOT.parent
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
sys.path.insert(0,str(ROOT/'src'))
from stage3 import run
print('Project Python:',sys.executable)
report=run()
assert report['all_passed']'''),
md('''## Actual verification results

The parameter counts below are measured from instantiated models and include 64 fixed PAD entries. UNK gradients/updates come from actual 2% masking on training examples. The 24-example balanced overfit check uses a separate fresh model, plain cross-entropy, learning rate 0.01 and disables dropout/masking to isolate learning ability. Its accuracy is not a generalisation estimate. All diagnostic weights are discarded; no checkpoint is saved.'''),
code('''import pandas as pd
from IPython.display import display
display(pd.Series(report['checks'],name='passed').to_frame())
display(pd.DataFrame({name:{'parameters':r['parameters'],'padding_difference':r['padding_max_abs_difference'],
    'UNK_gradient_norm':r['unk_gradient_norm'],'UNK_update_norm':r['unk_update_norm'],
    'tiny_subset_accuracy':r['overfit_diagnostic']['training_subset_accuracy'],
    'tiny_initial_loss':r['overfit_diagnostic']['initial_loss'],'tiny_final_loss':r['overfit_diagnostic']['final_loss'],
    'tiny_steps':r['overfit_diagnostic']['steps']} for name,r in report['models'].items()}).T)
display(report['masking'])'''),
md('''## CPU timing

Fresh models, one at a time: five warm-up then twenty timed training batches, batch size 32, two CPU threads, float32 and no loader workers. Timing includes batch construction, masking, forward/backward, gradient clipping and Adam updates. It excludes initialisation, validation and checkpoints. Epoch estimates use 313 batches and are approximate; laptop load and batch lengths vary.'''),
code('''display(report['environment'])
display(pd.DataFrame({name:{k:v for k,v in r['cpu_timing'].items() if k!='batch_seconds'} for name,r in report['models'].items()}).T)
assert report['diagnostic_weights_discarded'] and not report['weights_saved']
assert not report['baseline_training_started'] and not report['test_performance_accessed']
print('Stage 2 hashes preserved. Diagnostic weights discarded. Baseline training pending review.')'''),
md('''## Pending baseline plan and assistance record

See `STAGE3.md` for the proposed shared loss/optimiser/batch/early-stopping settings. Future training must call `fresh_model` and create a new optimiser, reset seeds and use only validation macro-F1 for early stopping. Test scoring remains deferred.

Codex generated and executed model, masking and diagnostic code and prepared working result notes. This includes diagnostic training on training-only subsets. No baseline training, tuning, held-out performance evaluation, pretrained weights or assignment report was produced. Any later disclosure must accurately describe this assistance under applicable requirements.''')],metadata={'kernelspec':{'name':'python3','display_name':'ODL project Python','language':'python'}})
km=KernelManager(kernel_name='python3')
km._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='ODL project Python',language='python')
try:
    NotebookClient(nb,km=km,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True)
except Exception:
    nbf.write(nb,out/'failed_execution.ipynb'); raise
nbf.write(nb,path)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'executed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'code_cells':3,'cell_errors':0,
        'previous_snapshot':str(snapshot) if snapshot else None,'notebook_sha256':sha(path),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/models.py',ROOT/'src/stage3.py',Path(__file__)]},
        'output_hashes':{p.name:sha(p) for p in out.glob('*.json') if p.name!='notebook_execution.json'}}
(out/'notebook_execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
