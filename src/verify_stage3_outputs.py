"""Verify saved evidence without constructing models or running more diagnostics."""
import hashlib
import json
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/stage3'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=load(OUT/'notebook_execution.json')
report=load(OUT/'verification.json')
preservation=load(OUT/'stage2_preservation.json')
assert preservation['before']==preservation['after']
assert all(sha(ROOT/p)==h for p,h in preservation['before'].items())
assert all(sha(ROOT/p)==h for p,h in record['source_hashes'].items())
assert all(sha(OUT/p)==h for p,h in record['output_hashes'].items())
notebook=ROOT/'notebooks/03_model_verification.ipynb'
assert sha(notebook)==record['notebook_sha256']
nb=nbformat.read(notebook,as_version=4)
cells=[c for c in nb.cells if c.cell_type=='code']
assert len(cells)==record['code_cells']==3
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in cells)
assert report['all_passed'] and all(report['checks'].values())
assert not report['weights_saved'] and report['diagnostic_weights_discarded']
assert not report['baseline_training_started'] and not report['test_performance_accessed']
assert not any(OUT.glob('*.pt')) and not any(OUT.glob('*.pth'))
for name,r in report['models'].items():
    timing=r['cpu_timing']
    assert len(timing['batch_seconds'])==20 and timing['warmup_batches']==5
    assert abs(sum(timing['batch_seconds'])/20-timing['mean_batch_seconds'])<1e-10
    assert abs(timing['mean_batch_seconds']*313-timing['estimated_313_batch_training_epoch_seconds'])<1e-8
result={'saved_evidence_verified':True,'stage2_preserved':True,'passed_diagnostic_checks':len(report['checks']),
        'executed_code_cells':len(cells),'scope':'read-only evidence verification; no new training',
        'verification_script_sha256':sha(Path(__file__))}
(OUT/'saved_evidence_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
