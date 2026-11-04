"""Selection protocol, interruption/reuse, and flat physical-output identity."""
import json
from types import SimpleNamespace

import pytest
import torch

from uav_arena.participants.jepa import selection_queue as q
from uav_arena.participants.jepa.model import JEPAPredictor
from uav_arena.participants.jepa.objective import StateReadout


@pytest.fixture
def case(tmp_path):
    checkpoint = tmp_path/'model.pt'
    job = dict(name='test', variant='full', config='G1', history=1, steps=23000,
               parameters=1288841, checkpoint=checkpoint, result=tmp_path/'result.json',
               training=['train'], evaluation=['evaluate'])
    grid = dict(source_sha256={'train.py': 'source'},
                data_manifest_sha256={'explore_v2': 'train', 'explore_v2_val': 'val'},
                validation_decisions_per_latency=10)
    result = dict(config=dict(stride='1', horizon='15',
        predictor='uav_arena.participants.jepa.model:make',
        predictor_options=json.dumps(dict(checkpoint=str(checkpoint))),
        seeds='299000:299100', data='data/uav_arena/datasets/explore_v2_val'),
        episodes=[f'episode_{s}.npz' for s in range(299000,299100)], selection_score=.02,
        latencies={f'{d}ms': dict(decisions=10, payload=[.02]*15, payload_mean=.02,
                   uav=[.03]*15, uav_mean=.03) for d in (0,50,100,200)})
    return tmp_path, job, grid, result


def test_validation_accepts_full_protocol(case):
    _, job, grid, result = case
    assert q.validate_result(result, job, grid) == .02


@pytest.mark.parametrize('fault', ['stride','episodes','latencies','decisions','weights','nan','mean'])
def test_validation_rejects_partial_or_mismatched_outputs(case, fault):
    _, job, grid, result = case
    if fault == 'stride': result['config']['stride'] = '10'
    if fault == 'episodes': result['episodes'].pop()
    if fault == 'latencies': del result['latencies']['200ms']
    if fault == 'decisions': result['latencies']['0ms']['decisions'] = 9
    if fault == 'weights': result['config']['predictor_options'] = '{"checkpoint":"wrong.pt"}'
    if fault == 'nan': result['selection_score'] = float('nan')
    if fault == 'mean': result['latencies']['0ms']['payload_mean'] = .04
    with pytest.raises(ValueError): q.validate_result(result, job, grid)


def test_queue_generates_fresh_fixed_seed_commands_without_stride(tmp_path):
    grid = dict(configs=[dict(id=c, history=h, steps=s,
        variants={'full': dict(checkpoint=f'data/uav_arena/participants/jepa/phase_b_select_full_{c.lower()}_s0.pt',
        result=f'data/uav_arena/results/phase_b_select_full_{c.lower()}_s0_openloop.json',
        parameters={'total': 1})}) for c,h,s in q.CONFIGS])
    queue = list(q.jobs(grid, 'full', tmp_path))
    assert len(queue) == 6
    assert [(j['config'], j['history'], j['steps']) for j in queue[-2:]] == [
        ('G5', 1, 60000), ('G6', 4, 60000)]
    for job in queue:
        command = job['training']
        assert command[command.index('--steps')+1] == str(job['steps'])
        assert command[command.index('--history')+1] == str(job['history'])
        assert '--resume' not in command and command[command.index('--seed')+1] == '0'
        assert command[command.index('--episodes')+1] == '2000'
        assert '--stride' not in job['evaluation']


def test_resume_checks_recipe_source_and_data(case):
    _, job, grid, _ = case
    pack = dict(kind='jepa_training_resume_v1', training={**q.expected_training(job),
        'data':'data/uav_arena/datasets/explore_v2'}, source_sha256=grid['source_sha256'],
        dataset_manifest_sha256='train', step=100)
    q.validate_resume(pack, job, grid)
    for key, bad in [('training', {**pack['training'], 'seed':1}),
                     ('source_sha256', {}), ('dataset_manifest_sha256','other'), ('step',23001)]:
        changed = {**pack, key:bad}
        with pytest.raises(ValueError): q.validate_resume(changed, job, grid)


def test_finished_job_is_checked_then_reused(case, monkeypatch):
    root, job, grid, result = case
    job['checkpoint'].write_text('weights')
    q.write(job['result'], result)
    receipt = {k:grid[k] for k in ('source_sha256','data_manifest_sha256')}
    receipt.update(checkpoint_sha256=q.digest(job['checkpoint']),result_sha256=q.digest(job['result']))
    q.write(root/'data/uav_arena/results/jepa/test_queue.json',receipt)
    monkeypatch.setattr(q,'training_complete',lambda *_:True)
    monkeypatch.setattr(q,'run_command',lambda *_:pytest.fail('completed job reran'))
    assert q.run_one(job,grid,root)==.02
    job['checkpoint'].write_text('changed')
    with pytest.raises(ValueError,match='weights'): q.run_one(job,grid,root)


def test_partial_training_resumes_same_checkpoint(case, monkeypatch):
    root,job,grid,_=case
    job['checkpoint'].with_suffix('.resume.pt').write_text('complete state')
    monkeypatch.setattr(q,'training_complete',lambda *_:False)
    monkeypatch.setattr(q.torch,'load',lambda *a,**k:{})
    monkeypatch.setattr(q,'validate_resume',lambda *_:None)
    commands=[]
    monkeypatch.setattr(q,'run_command',lambda cmd,log:commands.append(cmd))
    assert q.run_one(job,grid,root) is None
    assert '--resume' in commands[0] and '--steps' not in commands[0]
    assert q.read(root/'data/uav_arena/results/jepa/test_queue.json')['stage']=='paused'


def test_subprocess_error_stops_and_records_failure(case,monkeypatch):
    root,job,grid,_=case
    monkeypatch.setattr(q,'training_complete',lambda *_:False)
    def fail(*_): raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(q,'run_command',fail)
    with pytest.raises(RuntimeError,match='CUDA'): q.run_one(job,grid,root)
    receipt=q.read(root/'data/uav_arena/results/jepa/test_queue.json')
    assert receipt['stage']=='failed' and 'CUDA' in receipt['error']


def test_result_without_bound_weights_is_not_overwritten(case,monkeypatch):
    root,job,grid,result=case
    job['checkpoint'].write_text('weights')
    q.write(job['result'],result)
    before=job['result'].read_bytes()
    monkeypatch.setattr(q,'training_complete',lambda *_:True)
    monkeypatch.setattr(q,'run_command',lambda *_:pytest.fail('unbound result overwritten'))
    with pytest.raises(ValueError,match='Unbound'): q.run_one(job,grid,root)
    assert job['result'].read_bytes()==before


def test_final_checkpoint_requires_matching_weights_and_optimizer_step(case):
    _,job,grid,_=case
    info=dict(training=q.expected_training(job),parameters=job['parameters'],
        selected_seeds=[200000,201999],log=[{'step':23000}],
        source_sha256=grid['source_sha256'],dataset_manifest_sha256='train',config={})
    weights={'weight':torch.tensor([1.])}
    q.write(job['checkpoint'].with_suffix('.json'),info)
    torch.save(dict(config={},training_info=info,model=weights),job['checkpoint'])
    resume=dict(kind='jepa_training_resume_v1',training={**q.expected_training(job),
        'data':'data/uav_arena/datasets/explore_v2'},source_sha256=grid['source_sha256'],
        dataset_manifest_sha256='train',step=23000,scheduler={'last_epoch':23000},
        optimizer={'state':{0:{'step':torch.tensor(23000.)}}},model=weights)
    path=job['checkpoint'].with_suffix('.resume.pt')
    torch.save(resume,path)
    assert q.training_complete(job,grid)
    resume['optimizer']['state'][0]['step']=torch.tensor(100.)
    torch.save(resume,path)
    with pytest.raises(ValueError,match='optimizer'): q.training_complete(job,grid)
    resume['optimizer']['state'][0]['step']=torch.tensor(23000.)
    resume['model']={'weight':torch.tensor([2.])}
    torch.save(resume,path)
    with pytest.raises(ValueError,match='weights differ'): q.training_complete(job,grid)


def test_lock_rejects_second_queue_and_releases(tmp_path):
    with q.queue_lock(tmp_path/'queue.lock'):
        with pytest.raises(RuntimeError,match='active'):
            with q.queue_lock(tmp_path/'queue.lock'): pass
    with q.queue_lock(tmp_path/'queue.lock'): pass


def test_flat_readout_and_adapter_preserve_aircraft_order_with_physical_scaling():
    head=StateReadout(32,'flat')
    with torch.no_grad():
        head.flat[-1].weight.zero_()
        head.flat[-1].bias.copy_(torch.arange(15.))
    predictor=JEPAPredictor.__new__(JEPAPredictor)
    predictor.device=torch.device('cpu')
    predictor.model=SimpleNamespace(readout=head,
        rollout=lambda z,recent,times,points,age:torch.zeros(len(points),points.shape[1],1,32))
    predictor.stats=dict(state_std=torch.arange(1.,52.),state_mean=torch.zeros(51))
    predictor.context=dict(z=torch.zeros(1,1,1,32),recent=torch.zeros(1,4,3,3),
        times=torch.zeros(1,4),origin=torch.tensor([[10.,20.,30.]]),age=torch.zeros(1))
    output=predictor(None,torch.zeros(1,2,15,3,3))
    expected=(torch.arange(9.)*torch.arange(1.,10.)).reshape(3,3)+predictor.context['origin']
    torch.testing.assert_close(output['uav'][0,0,0],expected)
    torch.testing.assert_close(output['payload'][0,0,0],
        torch.arange(9.,12.)*torch.arange(46.,49.)+predictor.context['origin'][0])
    torch.testing.assert_close(output['payload_vel'][0,0,0],torch.arange(12.,15.)*torch.arange(49.,52.))


@pytest.fixture
def six_grid():
    return dict(status='approved_ready', common=q.COMMON.copy(),
        training_episodes=2000, validation_stride=1, selection_seed=0, horizon=15,
        extension_approval_url=q.EXTENSION_APPROVAL, hard_step_cap=60000,
        source_sha256={}, data_manifest_sha256={}, flat_preflight={'cleared': True},
        configs=[dict(id=c, history=h, steps=s, variants={v:dict(
            checkpoint=f'data/uav_arena/participants/jepa/phase_b_select_{v}_{c.lower()}_s0.pt',
            result=f'data/uav_arena/results/phase_b_select_{v}_{c.lower()}_s0_openloop.json',
            parameters={'total':1}) for v in ('full','no_cross','flat')})
            for c,h,s in q.CONFIGS])


@pytest.mark.parametrize('fault', ['approval','cap','extra_steps','missing_candidate'])
def test_six_grid_rejects_unapproved_or_changed_extension(tmp_path,six_grid,fault):
    q.check_grid(six_grid,tmp_path)
    if fault == 'approval': del six_grid['extension_approval_url']
    if fault == 'cap': six_grid['hard_step_cap'] = 80000
    if fault == 'extra_steps': six_grid['configs'][-1]['steps'] = 80000
    if fault == 'missing_candidate': six_grid['configs'].pop()
    with pytest.raises(ValueError): q.check_grid(six_grid,tmp_path)


def test_extension_cannot_resume_a_40000_step_schedule(case):
    _,job,grid,_=case
    old=q.expected_training(job)
    job.update(config='G5',steps=60000)
    pack=dict(kind='jepa_training_resume_v1',training={**old,
        'data':'data/uav_arena/datasets/explore_v2'},step=40000,
        source_sha256=grid['source_sha256'],dataset_manifest_sha256='train')
    with pytest.raises(ValueError,match='configuration mismatch'):
        q.validate_resume(pack,job,grid)


def test_default_queue_is_dry_run(tmp_path,six_grid,monkeypatch,capsys):
    q.write(tmp_path/q.GRID,six_grid)
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(q,'ROOT',tmp_path)
    monkeypatch.setattr(q.sys,'argv',['selection_queue','--variant','full'])
    monkeypatch.setattr(q,'run_command',lambda *_:pytest.fail('dry run launched command'))
    monkeypatch.setattr(q,'run_one',lambda *_:pytest.fail('dry run changed receipt'))
    q.main()
    output=json.loads(capsys.readouterr().out)
    assert not output['launch'] and len(output['runs'])==6
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('variant',['full','no_cross','flat'])
@pytest.mark.parametrize('scores,winner',[
    ([.09,.08,.07,.06,.06,.05],'G6'),
    ([.09,.08,.07,.06,.06,.06],'G4')])
def test_final_selection_uses_all_six_raw_scores_and_stable_ties(
        tmp_path,six_grid,monkeypatch,variant,scores,winner):
    q.write(tmp_path/q.GRID,six_grid)
    monkeypatch.setattr(q,'ROOT',tmp_path)
    monkeypatch.setattr(q.sys,'argv',['selection_queue','--variant',variant,'--run'])
    monkeypatch.setattr(q,'ensure_idle',lambda:None)
    seen=[]
    def completed(job,*_):
        seen.append(job['config'])
        job['checkpoint'].parent.mkdir(parents=True,exist_ok=True)
        job['checkpoint'].write_text('test weights')
        return scores[int(job['config'][1:])-1]
    monkeypatch.setattr(q,'run_one',completed)
    monkeypatch.setattr(q,'run_command',lambda *_:pytest.fail('test launched real training'))
    q.main()
    result=q.read(tmp_path/f'data/uav_arena/results/jepa/phase_b_{variant}_selection.json')
    assert seen==result['completed_configs']==['G1','G2','G3','G4','G5','G6']
    assert result['selected']['config']==winner
    assert result['selection_complete'] and not result['selected_is_provisional']
    assert not result['grid_extension_review_required'] and result['hard_step_cap']==60000
    assert not result['formal_training_started']
