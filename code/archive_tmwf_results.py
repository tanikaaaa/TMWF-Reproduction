"""Archive existing read-only TMWF evaluators; --check never launches evaluators.

Logs contain merged, unmodified stdout/stderr bytes and are authoritative.
Metric extraction is deliberately omitted: no inferred or fabricated values.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/reproducibility'
F_CHECKPOINT = 'outputs/appendix_f/full/run_97g8i7wd/epoch_050.pth'
H_CHECKPOINT = 'outputs/appendix_h/full/run_8mrp3k_2/epoch_050.pth'
SPECS = (
    ('appendix_f_table7', 'evaluate_appendix_f.py', (), F_CHECKPOINT,
     'PASS: exactly 5000 samples per dataset; checkpoint and model state unchanged. No files written.',
     'Appendix F / Table 7 (complete paper p.20), TMWF rows'),
    ('appendix_h_ratio_real_table9', 'evaluate_appendix_h.py', ('--evaluate',), H_CHECKPOINT,
     'PASS: 6000 component predictions; 3000 samples per mixed set; model parameters/buffers, checkpoint and source dataset hashes unchanged.',
     'Appendix H / Table 9, TMWF: original metric rounding preserved.'),
    ('appendix_h_delay_table9', 'evaluate_appendix_h_delay.py', ('--evaluate',), H_CHECKPOINT,
     'PASS: 3000 predictions; checkpoint, model parameters/buffers, datasets and generation sources unchanged.',
     'Appendix H / Table 9 mixed delay-synthetic: 3000 samples.'),
)
DELAY_NOTE = ('Copy-isolated reproduction of the released TMWF delay-merging algorithm; '
              'NOT literal released-code execution. Independent per-occurrence copies; '
              'original delay scheduling and existing timestamp reversals preserved.')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_output():
    for path in (OUTPUT,) + tuple(OUTPUT.parents):
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f'Unsafe output-directory path: {path}')
    ancestor = OUTPUT
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(f'Output ancestor is not writable: {ancestor}')


def source_constants(path):
    source = path.read_text()
    compile(source, str(path), 'exec')  # No bytecode files or module execution.
    tree = ast.parse(source)
    strings = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return tree, strings


def model_config(tree):
    value = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'MODEL' for t in n.targets))
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != 'dict':
        raise ValueError('Unsupported MODEL declaration')
    return {k.arg: ast.literal_eval(k.value) for k in value.keywords}


def inventory():
    """Evaluation inputs only; never hash multi-GB training datasets here."""
    f_dirs = [f'dataset/generated_appendix_f/test_{t}tab' for t in (2,4,6)]
    h_dirs = [f'dataset/generated_appendix_h/ratio_test_{t}tab' for t in (2,3,4)]
    d_dirs = [f'dataset/generated_appendix_h_delay/copy_isolated/delay_test_{t}tab' for t in (2,3,4)]
    groups = [f_dirs, h_dirs + [f'dataset/tbb realworld/{t}tabs' for t in (2,3,4)],
              d_dirs + h_dirs + ['dataset/tbb single tab/test']]
    all_files = set()
    for group in groups:
        for relative in group:
            path = ROOT / relative
            if path.is_file():
                all_files.add(relative)
                continue
            if not path.is_dir():
                raise FileNotFoundError(path)
            is_f = 'generated_appendix_f/' in relative
            last = 5 if is_f else 1
            expected = {f'COCO_test_{i}' for i in range(last+1)} | {f'BAPM_test_{last}.npz'}
            expected |= {f'test_{i}_annotations.json' for i in range(last+1)}
            if 'generated_appendix_h_delay/' in relative:
                expected.add('manifest.json')
            actual = {p.name for p in path.iterdir()}
            if actual != expected or not all((path/n).is_file() for n in expected):
                raise ValueError(f'Unexpected or missing dataset files: {path}')
            all_files.update(str((path/n).relative_to(ROOT)) for n in expected)
    return groups, all_files


def check():
    safe_output()
    groups, inputs = inventory()
    expected_model = dict(embed_dim=256, nhead=8, dim_feedforward=1024,
                          num_encoder_layers=2, num_decoder_layers=2, max_len=121,
                          num_queries=6, cls=51, dropout=0.1)
    code = set()
    for path in sorted((ROOT/'code').glob('*.py')):
        source_constants(path)
    for name in ('train_appendix_f.py','train_appendix_h.py','evaluate_appendix_f.py','evaluate_appendix_h.py'):
        tree,_ = source_constants(ROOT/'code'/name)
        if model_config(tree) != expected_model:
            raise ValueError(f'Model configuration mismatch: {name}')
    plans=[]
    for index,(name,script,flags,checkpoint,marker,heading) in enumerate(SPECS):
        path=ROOT/'code'/script
        tree,strings=source_constants(path)
        options={a.value for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument'
                 for a in n.args if isinstance(a,ast.Constant) and isinstance(a.value,str)}
        if flags and not set(flags).issubset(options):
            raise ValueError(f'Unsupported evaluator CLI: {script}')
        if not flags and options:
            raise ValueError('Appendix F CLI changed; review invocation')
        if marker not in strings or not any(heading in value for value in strings):
            raise ValueError(f'Completion contract changed: {script}')
        if script != 'evaluate_appendix_h_delay.py' and checkpoint not in strings:
            raise ValueError(f'Checkpoint selection changed: {script}')
        if script == 'evaluate_appendix_h_delay.py':
            if not any(isinstance(n,ast.ImportFrom) and n.module=='evaluate_appendix_h'
                       and any(a.name=='CHECKPOINT' for a in n.names) for n in tree.body):
                raise ValueError('Delay checkpoint import changed')
        if not (ROOT/checkpoint).is_file():
            raise FileNotFoundError(ROOT/checkpoint)
        plans.append(dict(evaluator=name, command=[sys.executable,'-B','-u',str(path),*flags],
                          checkpoint_path=str(ROOT/checkpoint),dataset_references=groups[index],
                          source_script=str(path),expected_markers=[marker,heading],
                          execution_status='not_started',start_timestamp=None,end_timestamp=None,
                          exit_code=None,log_path=None,log_sha256=None,
                          source_script_sha256=None,checkpoint_sha256=None))
    # Source snapshots include dependencies; imports are not executed in --check.
    code={str(p.relative_to(ROOT)) for p in (ROOT/'code').glob('*.py')}
    protected=inputs | code | {F_CHECKPOINT,H_CHECKPOINT}
    print('CHECK PASS: syntax, static MODEL/CLI/checkpoint configuration, input inventories, '
          'checkpoint existence and output-path safety. No evaluator launched; no files written.',flush=True)
    for plan in plans: print(f"{plan['evaluator']}: {plan['command']}",flush=True)
    print('Checkpoint contents and full dataset integrity are checked by the existing evaluators during --archive.')
    return plans,protected


def write_json(path, payload):
    # Publish atomically without replacement; each report name is written once.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         prefix='.report_',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def archive(plans, protected):
    safe_output()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory=OUTPUT/f'{stamp}_{uuid.uuid4().hex}'
    directory.mkdir(exist_ok=False)
    summary=dict(start_timestamp=now(),end_timestamp=None,execution_status='running',
                 cwd=str(ROOT),python_executable=sys.executable,
                 environment=dict(python=platform.python_version(),platform=platform.platform(),
                                  numpy=version('numpy'),torch=version('torch'),
                                  mps_fallback=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')),
                 delay_protocol=DELAY_NOTE,metrics=None,
                 metric_policy='No metric parsing. Complete raw logs are authoritative.',
                 log_format='stdout and stderr merged into one binary log per evaluator',
                 hash_policy='One pre/post hash per unique evaluation input, source script and checkpoint; training datasets excluded.',
                 evaluators=plans)
    before={}; interrupted=False
    write_json(directory/'plan.json',summary)
    try:
        before={name:sha256(ROOT/name) for name in sorted(protected)}
        summary['hashes_before']=before
        for plan in plans:
            log=directory/f"{stamp}_{plan['evaluator']}.log"
            plan.update(start_timestamp=now(),log_path=str(log),
                        source_script_sha256=before[str(Path(plan['source_script']).relative_to(ROOT))],
                        checkpoint_sha256=before[str(Path(plan['checkpoint_path']).relative_to(ROOT))])
            process=None
            try:
                print(f"Running {plan['evaluator']}; complete output: {log}",flush=True)
                with log.open('xb') as stream:
                    process=subprocess.Popen(plan['command'],cwd=ROOT,stdout=stream,
                                             stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    try:
                        plan['exit_code']=process.wait()
                    except BaseException:
                        if process.poll() is None:
                            process.terminate()
                            try: process.wait(timeout=10)
                            except subprocess.TimeoutExpired: process.kill(); process.wait()
                        plan['exit_code']=process.returncode
                        raise
                    finally:
                        stream.flush(); os.fsync(stream.fileno())
                text=log.read_bytes().decode('utf-8',errors='replace')
                plan['markers_present']={m:m in text for m in plan['expected_markers']}
                good=plan['exit_code']==0 and all(plan['markers_present'].values())
                plan['execution_status']='completed_pending_integrity' if good else 'failed'
            except BaseException as exc:
                plan['execution_status']='failed'
                plan['error']=f'{type(exc).__name__}: {exc}'
                interrupted=isinstance(exc,(KeyboardInterrupt,SystemExit))
            finally:
                plan['end_timestamp']=now()
                if log.is_file(): plan['log_sha256']=sha256(log)
                write_json(directory/f"{plan['evaluator']}.json",plan)
            if interrupted: break
    except BaseException as exc:
        summary['error']=f'{type(exc).__name__}: {exc}'
    finally:
        try:
            _,current=inventory()
            after={name:sha256(ROOT/name) for name in sorted(protected)}
            summary['hashes_after']=after
            # Compare file inventory as well as content (detect new/missing dataset files).
            original_data={n for n in protected if n.startswith('dataset/')}
            unchanged=bool(before) and after==before and current==original_data
            summary['integrity_unchanged']=unchanged
        except Exception as exc:
            unchanged=False
            summary['integrity_error']=f'{type(exc).__name__}: {exc}'
        for plan in plans:
            if plan['execution_status']=='completed_pending_integrity':
                plan['execution_status']='success' if unchanged else 'failed_integrity'
        summary['end_timestamp']=now()
        summary['execution_status']='success' if unchanged and all(p['execution_status']=='success' for p in plans) else 'failed'
        write_json(directory/'summary.json',summary)
        print(f"Archive {summary['execution_status']}: {directory/'summary.json'}",flush=True)
    return 0 if summary['execution_status']=='success' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--check',action='store_true')
    modes.add_argument('--archive',action='store_true')
    args=parser.parse_args()
    try:
        plans,protected=check()
        return archive(plans,protected) if args.archive else 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__=='__main__':
    raise SystemExit(main())
