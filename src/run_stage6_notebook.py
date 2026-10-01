"""Create/execute Stage 6 diagnostics in the project venv, preserving prior runs."""
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
out=ROOT/'results/stage6'; out.mkdir(parents=True,exist_ok=True)
path=ROOT/'notebooks/06_final_test_evaluation.ipynb'
previous=[p for p in out.rglob('*') if p.is_file()]+([path] if path.exists() else [])
snapshot=None
if previous:
    snapshot=ROOT/'reviews'/('stage6_before_rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for p in previous:
        target=snapshot/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
runtime=ROOT/'.jupyter-runtime/stage6'; runtime.mkdir(parents=True,exist_ok=True)
os.environ.update(JUPYTER_RUNTIME_DIR=str(runtime),IPYTHONDIR=str(runtime/'ipython'),MPLCONFIGDIR=str(runtime/'matplotlib'))
md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook(cells=[
md("""# Stage 6: final frozen held-out test evaluation

All six preselected checkpoints: CNN and GRU, seeds42/43/44. Frozen protocol, source, checkpoint, split and label hashes are verified before any forward pass. No retraining, tuning, masking, threshold changes or seed selection. Evaluation mode and inference_mode are used. Test metrics are final descriptive analysis only.

Report each seed, plus arithmetic mean and sample SD (ddof=1). Repeated predictions of the same 2,141 tweets are not independent examples. Small example sample is fixed before scoring: seed42 only, one correct and one incorrect example per true class per model, selected by smallest SHA256(2026|model|source_row). No keyword search or post-result sample replacement.

See STAGE6.md for final working findings and marking-scheme checklist. This notebook is not the assignment report."""),
code("""from pathlib import Path
import sys
ROOT=Path.cwd().resolve()
if not (ROOT/'src/stage6.py').exists(): ROOT=ROOT.parent
assert Path(sys.prefix).resolve()==(ROOT/'.venv').resolve()
sys.path.insert(0,str(ROOT/'src'))
from stage6 import run
summary=run()
assert summary['all_saved_metrics_verified'] and summary['previous_artifacts_preserved']"""),
code("""import pandas as pd
from IPython.display import display, Image
display(pd.read_csv(ROOT/'results/stage6/checkpoint_comparison.csv'))
display(pd.read_csv(ROOT/'results/stage6/model_summary.csv'))
display(pd.read_csv(ROOT/'results/stage6/per_class_metrics.csv'))
print('Evaluation wall seconds before plots:',summary['evaluation_wall_seconds_before_plots'])"""),
code("""display(Image(filename=str(ROOT/'results/stage6/test_confusion_matrices.png')))
display(Image(filename=str(ROOT/'results/stage6/test_model_comparison.png')))
display(Image(filename=str(ROOT/'results/stage6/validation_test_comparison.png')))
sample=pd.read_csv(ROOT/'results/stage6/example_sample.csv',dtype={'source_row':str,'tweet_id':str})
display(sample[['model','source_row','true_label','predicted_label','correct','text','unseen_tokens_json']])
print('Final evaluation complete. No model choices changed after test access.')"""),
md("""Metric CSV/JSON files and per-checkpoint prediction CSV/NPZ files provide traceability to source rows, original tweet IDs, true labels, probabilities and logits. Every matrix is seed-specific. Sample examples are illustrative, not a representative error-rate study. Human interpretations and remaining assignment work are in STAGE6.md. Codex generated and executed final evaluation code and working analysis; no assignment report or AI acknowledgement is written.""")],metadata={'kernelspec':{'name':'python3','display_name':'ODL project Python','language':'python'}})
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
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/models.py',ROOT/'src/stage6.py',Path(__file__)]},
        'output_hashes':{p.name:sha(p) for p in out.glob('*.json') if p.name!='notebook_execution.json'}}
(out/'notebook_execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
