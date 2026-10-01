"""Create/execute Stage 4 diagnostics in the project venv, preserving prior runs."""
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
out=ROOT/'results/stage4'; out.mkdir(parents=True,exist_ok=True)
path=ROOT/'notebooks/04_baseline_training.ipynb'
previous=[p for p in out.glob('*') if p.is_file()]+([path] if path.exists() else [])
snapshot=None
if previous:
    snapshot=ROOT/'reviews'/('stage4_before_rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for p in previous:
        target=snapshot/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
runtime=ROOT/'.jupyter-runtime/stage4'; runtime.mkdir(parents=True,exist_ok=True)
os.environ.update(JUPYTER_RUNTIME_DIR=str(runtime),IPYTHONDIR=str(runtime/'ipython'),MPLCONFIGDIR=str(runtime/'matplotlib'))
md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md("""# Stage 4: fixed baseline training

Custom CNN and GRU, random initialisation. Shared seed42, weighted CE from training labels, Adam 0.001, batch32, dropout0.3, clipping1.0, max20 epochs, validation macro-F1 early stopping patience3 (significant change >0.0001). Highest observed validation F1 checkpoint is saved, earliest exact tie. Stage 2 and Stage 3 are read-only; test files are hashed for preservation but never loaded for model evaluation. No tuning is performed.

Training metrics aggregate predictions made during optimisation with masking and dropout active; validation metrics use unmasked inputs in eval mode. Weighted epoch losses use summed weighted losses divided by summed target weights. These curves are not directly comparable estimates on identically processed inputs.

The primary rubric assigns 30 marks to model building and 10 to tuning; this run is the baseline, not completed tuning or final test evaluation. See STAGE4.md for source references, results, interpretation and proposed next steps."""),
code("""from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if not (ROOT/'src/stage4.py').exists(): ROOT=ROOT.parent
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
sys.path.insert(0,str(ROOT/'src'))
from stage4 import run
summary=run()
assert summary['previous_outputs_preserved'] and not summary['test_evaluated']"""),
code("""import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(ROOT/'results/stage4/baseline_comparison.csv'))
for kind,r in summary['models'].items():
 print(kind,'best epoch',r['best_epoch'],'reload predictions/logits exact:',r['reload_predictions_exact'],r['reload_logits_exact'])
 display(pd.DataFrame(r['per_class']).T)
print('Total training and validation seconds:',summary['total_model_training_validation_seconds'])"""),
code("""display(Image(filename=str(ROOT/'results/stage4/learning_curves.png')))
display(Image(filename=str(ROOT/'results/stage4/validation_confusion_matrices.png')))
print('Baseline complete. Test evaluation and tuning have not started.')"""),
md("""Checkpoints, per-epoch logs, validation predictions, class reports, raw confusion matrices, figures and hash evidence are in results/stage4/. Re-running starts fresh and archives the previous Stage 4 outputs. Validation is used for checkpoint selection, so these metrics are not an untouched final estimate. Codex generated and executed baseline training and analysis code; no assignment report or AI acknowledgement is written here.""")],metadata={'kernelspec':{'name':'python3','display_name':'ODL project Python','language':'python'}})
km=KernelManager(kernel_name='python3')
km._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='ODL project Python',language='python')
try:
    NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True)
except Exception:
    nbf.write(nb,out/'failed_execution.ipynb'); raise
nbf.write(nb,path)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'executed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'code_cells':3,'cell_errors':0,
        'previous_snapshot':str(snapshot) if snapshot else None,'notebook_sha256':sha(path),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/models.py',ROOT/'src/stage4.py',Path(__file__)]},
        'output_hashes':{p.name:sha(p) for p in out.glob('*.json') if p.name!='notebook_execution.json'}}
(out/'notebook_execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
