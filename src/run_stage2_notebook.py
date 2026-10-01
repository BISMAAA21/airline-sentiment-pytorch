"""Execute Stage 2 in the project venv; snapshot previous Stage 2 outputs first."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]
assert Path(sys.prefix).resolve() == (ROOT/'.venv').resolve()
runtime = ROOT/'.jupyter-runtime/stage2'
runtime.mkdir(parents=True,exist_ok=True)
os.environ.update(JUPYTER_RUNTIME_DIR=str(runtime), IPYTHONDIR=str(runtime/'ipython'),MPLCONFIGDIR=str(runtime/'matplotlib'))
path = ROOT/'notebooks/02_preprocessing_and_split.ipynb'
outputs = [p for folder in ['results/stage2','data/processed/stage2'] for p in (ROOT/folder).glob('*')
           if p.is_file() and p.name != 'notebook_execution.json']
old = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
snapshot = None
if outputs:
    snapshot = ROOT/'reviews'/('stage2_before_rerun_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    for p in [*outputs,path,*list((ROOT/'results/stage2').glob('notebook_execution.json'))]:
        target = snapshot/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    (snapshot/'manifest.json').write_text(json.dumps(old,indent=2),encoding='utf-8')
km = KernelManager(kernel_name='python3')
km._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                             display_name='ODL project Python',language='python')
nb = nbformat.read(path,as_version=4)
for c in nb.cells:
    if c.cell_type == 'code':
        c.outputs=[]
        c.execution_count=None
try:
    NotebookClient(nb,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute(cleanup_kc=True)
except Exception:
    nbformat.write(nb,ROOT/'results/stage2/failed_execution.ipynb')
    raise
nbformat.write(nb,path)
subprocess.run([sys.executable,str(ROOT/'src/verify_stage2.py')],check=True,capture_output=True,text=True)
outputs = [p for folder in ['results/stage2','data/processed/stage2'] for p in (ROOT/folder).glob('*')
           if p.is_file() and p.name != 'notebook_execution.json']
new = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
summary = json.loads((ROOT/'results/stage2/summary.json').read_text())
record = {'executed_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,
          'code_cells_executed':sum(c.cell_type=='code' for c in nb.cells),'cell_errors':0,
          'data_gate_status':summary['status'],'all_data_checks_passed':summary['all_checks_passed'],
          'prior_outputs_snapshot':str(snapshot) if snapshot else None,
          'repeat_artifacts_identical':new==old if old else None,
          'artifact_hashes':new,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT/'results/stage2/notebook_execution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
