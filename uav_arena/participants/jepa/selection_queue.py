"""Serial approved JEPA selection runs. Default is read-only; --run launches jobs."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
GRID = Path('data/uav_arena/results/jepa/phase_b_grid.json')
COMMON = dict(batch=128, lr=.001, dim=128, latent=64, enc_layers=2,
              pred_layers=4, heads=4, head_weight=20., sigreg_weight=.01,
              sigreg_projections=1024, sigreg_per_token=False, rollout_steps=15,
              stats_samples=5000, seed=0, workers=4, device='cuda')
CONFIGS = [('G1', 1, 23000), ('G2', 1, 40000), ('G3', 4, 23000), ('G4', 4, 40000)]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def check_grid(grid, root):
    checked(grid['status'] == 'approved_ready' and grid['common'] == COMMON,
            'Grid approval or fixed recipe mismatch')
    checked([(c['id'], c['history'], c['steps']) for c in grid['configs']] == CONFIGS,
            'Unexpected configurations')
    checked(grid['training_episodes'] == 2000 and grid['validation_stride'] == 1
            and grid['selection_seed'] == 0 and grid['horizon'] == 15,
            'Unexpected data budget or evaluation settings')
    for name, sha in grid['source_sha256'].items():
        checked(digest(root/'uav_arena/participants/jepa'/name) == sha,
                f'Training source changed: {name}')
    for directory, sha in grid['data_manifest_sha256'].items():
        checked(digest(root/'data/uav_arena/datasets'/directory/'manifest.json') == sha,
                f'Dataset manifest changed: {directory}')


def jobs(grid, variant, root):
    for config in grid['configs']:
        item = config['variants'][variant]
        name = f"phase_b_select_{variant}_{config['id'].lower()}_s0"
        checkpoint = root/f'data/uav_arena/participants/jepa/{name}.pt'
        result = root/f'data/uav_arena/results/{name}_openloop.json'
        checked(root/item['checkpoint'] == checkpoint and root/item['result'] == result,
                'Output paths do not match the approved run')
        training = [sys.executable, '-u', '-m', 'uav_arena.participants.jepa.train',
                    '--data', 'data/uav_arena/datasets/explore_v2', '--episodes', '2000',
                    '--history', str(config['history']), '--steps', str(config['steps']),
                    '--attn', variant]
        for key, value in COMMON.items():
            if not isinstance(value, bool):
                training += ['--' + key.replace('_', '-'), str(value)]
        training += ['--out', str(checkpoint)]
        evaluation = [sys.executable, '-u', '-m', 'uav_arena.openloop', '--predictor',
                      'uav_arena.participants.jepa.model:make', '--predictor-options',
                      json.dumps(dict(checkpoint=str(checkpoint), device='cuda')),
                      '--data', 'data/uav_arena/datasets/explore_v2_val',
                      '--seeds', '299000:299100', '--out', str(result)]
        yield dict(name=name, variant=variant, config=config['id'], history=config['history'],
                   steps=config['steps'], parameters=item['parameters']['total'],
                   checkpoint=checkpoint, result=result, training=training, evaluation=evaluation)


def expected_training(job):
    return {**COMMON, 'episodes': 2000, 'history': job['history'],
            'steps': job['steps'], 'attn': job['variant']}


def validate_resume(pack, job, grid):
    checked(pack.get('kind') == 'jepa_training_resume_v1', 'Not a complete checkpoint')
    checked(all(pack['training'].get(k) == v for k, v in expected_training(job).items()),
            'Resume configuration mismatch')
    checked(Path(pack['training']['data']).as_posix() == 'data/uav_arena/datasets/explore_v2',
            'Resume training dataset mismatch')
    checked(pack['source_sha256'] == grid['source_sha256'] and
            pack['dataset_manifest_sha256'] == grid['data_manifest_sha256']['explore_v2'],
            'Resume source or dataset mismatch')
    checked(0 < pack['step'] <= job['steps'], 'Invalid resume step')


def training_complete(job, grid):
    checkpoint = job['checkpoint']
    metadata = checkpoint.with_suffix('.json')
    if not checkpoint.exists():
        checked(not metadata.exists(), 'Metadata exists without final weights')
        return False
    checked(metadata.exists(), 'Final weights have no metadata')
    info = read(metadata)
    checked(all(info['training'].get(k) == v for k, v in expected_training(job).items()),
            'Completed training configuration mismatch')
    checked(info['parameters'] == job['parameters'] and
            info['selected_seeds'] == [200000, 201999] and
            info['log'][-1]['step'] == job['steps'], 'Incomplete or incorrect training')
    checked(info['source_sha256'] == grid['source_sha256'] and
            info['dataset_manifest_sha256'] == grid['data_manifest_sha256']['explore_v2'],
            'Completed source or data mismatch')
    pack = torch.load(checkpoint, map_location='cpu', weights_only=True)
    checked(pack['config'] == info['config'] and pack['training_info'] == info,
            'Weight metadata mismatch')
    checked(all(torch.isfinite(v).all().item() for v in pack['model'].values()),
            'Nonfinite final weights')
    resume = torch.load(checkpoint.with_suffix('.resume.pt'), map_location='cpu', weights_only=True)
    validate_resume(resume, job, grid)
    checked(resume['step'] == job['steps'] and resume['scheduler']['last_epoch'] == job['steps'],
            'Final optimizer/scheduler checkpoint incomplete')
    checked(bool(resume['optimizer']['state']) and all(
        int(value['step']) == job['steps'] for value in resume['optimizer']['state'].values()),
        'Final optimizer update count mismatch')
    checked(all(torch.equal(v, resume['model'][k]) for k, v in pack['model'].items()),
            'Final weights differ from complete checkpoint')
    return True


def validate_result(result, job, grid):
    config = result['config']
    checked(result['episodes'] == [f'episode_{s}.npz' for s in range(299000, 299100)],
            'Expected all 100 validation episodes')
    checked(int(config['stride']) == 1 and int(config['horizon']) == 15 and
            config['predictor'] == 'uav_arena.participants.jepa.model:make' and
            config['seeds'] == '299000:299100' and
            config['data'] == 'data/uav_arena/datasets/explore_v2_val',
            'Validation protocol mismatch')
    options = json.loads(config['predictor_options'])
    checked(Path(options['checkpoint']).resolve() == job['checkpoint'].resolve(),
            'Validation checkpoint mismatch')
    checked(set(result['latencies']) == {'0ms', '50ms', '100ms', '200ms'},
            'Missing latency')
    for row in result['latencies'].values():
        checked(row['decisions'] == grid['validation_decisions_per_latency'],
                'Incomplete validation decisions')
        for key in ('uav', 'payload'):
            checked(len(row[key]) == 15 and all(math.isfinite(v) and v >= 0 for v in row[key]),
                    'Invalid per-step errors')
            checked(math.isclose(sum(row[key])/15, row[key+'_mean'], abs_tol=1e-12),
                    'Mean error mismatch')
    score = result['selection_score']
    checked(math.isfinite(score) and score >= 0 and math.isclose(
        score, sum(r['payload_mean'] for r in result['latencies'].values())/4, abs_tol=1e-12),
        'Invalid selection score')
    return score


@contextlib.contextmanager
def queue_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another selection queue is active') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def ensure_idle():
    modules = {b'uav_arena.participants.jepa.train', b'uav_arena.participants.jepa.continue_train',
               b'uav_arena.openloop'}
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except OSError:
            continue
        if b'-m' in argv and modules.intersection(argv):
            raise RuntimeError(f'Existing training/evaluation process: {path.parent.name}')


def run_command(command, log):
    environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def run_one(job, grid, root):
    receipt_path = root/f"data/uav_arena/results/jepa/{job['name']}_queue.json"
    receipt = read(receipt_path) if receipt_path.exists() else {}
    identity = dict(source_sha256=grid['source_sha256'],
                    data_manifest_sha256=grid['data_manifest_sha256'])
    if receipt:
        checked(all(receipt.get(k) == v for k, v in identity.items()), 'Run identity changed')

    def status(stage, **values):
        receipt.update(identity, name=job['name'], stage=stage, updated_unix=time.time(), **values)
        write(receipt_path, receipt)
        print(json.dumps(dict(run=job['name'], stage=stage, **values)), flush=True)

    try:
        complete = training_complete(job, grid)
        if not complete:
            checked(not job['result'].exists(), 'Result exists without final weights')
            resume = job['checkpoint'].with_suffix('.resume.pt')
            command = job['training']
            if resume.exists():
                validate_resume(torch.load(resume, map_location='cpu', weights_only=True), job, grid)
                command = [sys.executable, '-u', '-m', 'uav_arena.participants.jepa.train',
                           '--resume', str(resume), '--out', str(job['checkpoint']),
                           '--device', 'cuda', '--workers', '4']
            status('training', command=command)
            run_command(command, root/f"data/uav_arena/logs/{job['name']}_train.log")
            if not training_complete(job, grid):
                status('paused')
                return None
        sha = digest(job['checkpoint'])
        if job['result'].exists():
            checked(receipt.get('checkpoint_sha256') == sha, 'Unbound or changed evaluation weights')
            if receipt.get('result_sha256'):
                checked(receipt['result_sha256'] == digest(job['result']), 'Evaluation result changed')
        else:
            status('validating', checkpoint_sha256=sha, command=job['evaluation'])
            run_command(job['evaluation'], root/f"data/uav_arena/logs/{job['name']}_val.log")
        score = validate_result(read(job['result']), job, grid)
        status('complete', checkpoint_sha256=sha, result_sha256=digest(job['result']),
               selection_score=score)
        return score
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('full', 'no_cross', 'flat'), default='full')
    parser.add_argument('--run', action='store_true', help='Launch all four runs for this variant')
    args = parser.parse_args()
    grid = read(ROOT/GRID)
    check_grid(grid, ROOT)
    checked(args.variant != 'flat' or grid['flat_preflight']['cleared'], 'Flat check unresolved')
    queue = list(jobs(grid, args.variant, ROOT))
    print(json.dumps(dict(variant=args.variant, runs=[j['name'] for j in queue], launch=args.run)), flush=True)
    if not args.run:
        return
    with queue_lock(ROOT/'data/uav_arena/results/jepa/phase_b_selection.lock'):
        ensure_idle()
        scores = []
        for job in queue:
            score = run_one(job, grid, ROOT)
            if score is None:
                return
            scores.append(dict(config=job['config'], history=job['history'], steps=job['steps'],
                               selection_score=score, checkpoint_sha256=digest(job['checkpoint'])))
        selected = min(scores, key=lambda row: row['selection_score'])
        write(ROOT/f'data/uav_arena/results/jepa/phase_b_{args.variant}_selection.json',
              dict(scores=scores, selected=selected, formal_training_started=False,
                   grid_extension_review_required=selected['steps'] == 40000))
        print(json.dumps(dict(selected=selected, grid_extension_review_required=selected['steps'] == 40000)), flush=True)


if __name__ == '__main__':
    main()
