"""Create/execute Stage 5 diagnostics in the project venv, preserving prior runs."""
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
out=ROOT/'results/stage5'; out.mkdir(parents=True,exist_ok=True)
path=ROOT/'notebooks/05_tuning_and_seed_variability.ipynb'
previous=[p for p in out.rglob('*') if p.is_file()]+([path] if path.exists() else [])
snapshot=None
if previous:
    snapshot=ROOT/'reviews'/('stage5_before_rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for p in previous:
        target=snapshot/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
runtime=ROOT/'.jupyter-runtime/stage5'; runtime.mkdir(parents=True,exist_ok=True)
os.environ.update(JUPYTER_RUNTIME_DIR=str(runtime),IPYTHONDIR=str(runtime/'ipython'),MPLCONFIGDIR=str(runtime/'matplotlib'))
md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md("""# Stage 5 tuning and seed variability

Two custom PyTorch models. Grid learning rate {0.001,0.0005} x dropout {0.3,0.5}. Reuse baseline seed42 only after exact saved configuration/source checks. Six fresh grid runs, then four fresh repeats with seeds43/44 at the selected configurations. Everything else remains fixed.

Selection before new training: highest seed42 validation macro-F1; exact ties use earlier best epoch then listed grid order. Report all three seeds and sample SD (ddof=1), never the luckiest seed. Validation selection is not an untouched final estimate. Test checkpoints/protocol are frozen at the end without test inference.

The primary rubric allocates ten marks to justified tuning; the brief Assignment 2 Task 2 permits grid search. Source references and interpretations are in STAGE5.md. This working notebook is not an assignment report."""),
code("""from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if not (ROOT/'src/stage5.py').exists(): ROOT=ROOT.parent
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
sys.path.insert(0,str(ROOT/'src'))
from stage5 import run
summary=run()
assert summary['previous_outputs_preserved'] and not summary['test_evaluated']"""),
code("""import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(ROOT/'results/stage5/tuning_comparison.csv'))
display(pd.read_csv(ROOT/'results/stage5/seed_comparison.csv'))
display(summary['variability'])
print('New training/validation seconds:',summary['new_training_validation_seconds'])
print('Baseline reuse and exact checkpoint reload checks passed.')"""),
code("""display(Image(filename=str(ROOT/'results/stage5/grid_comparison.png')))
display(Image(filename=str(ROOT/'results/stage5/seed_variability.png')))
print('Six selected checkpoints frozen. Test performance has not been accessed.')"""),
md("""Saved trial folders contain configurations, logs, checkpoints, validation logits, classification reports and runtime records. The reused baseline is explicitly identified and costs zero new training time. Fresh training starts from random weights, never baseline checkpoints. See frozen_test_protocol.json for exact paths/hashes of all six future test checkpoints. Repeats assess training randomness on the same split, not dataset/split uncertainty.

Codex generated and executed tuning and additional-seed training code and working analysis. No test evaluation or assignment report is produced. Any later assistance disclosure must describe this actual scope accurately.""")],metadata={'kernelspec':{'name':'python3','display_name':'ODL project Python','language':'python'}})
km=KernelManager(kernel_name='python3')
km._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='ODL project Python',language='python')
try:
    NotebookClient(nb,km=km,timeout=3600,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True)
except Exception:
    nbf.write(nb,out/'failed_execution.ipynb'); raise
nbf.write(nb,path)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record={'executed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,'code_cells':3,'cell_errors':0,
        'previous_snapshot':str(snapshot) if snapshot else None,'notebook_sha256':sha(path),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/models.py',ROOT/'src/stage5.py',Path(__file__)]},
        'output_hashes':{p.name:sha(p) for p in out.glob('*.json') if p.name!='notebook_execution.json'}}
(out/'notebook_execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
