import copy

import pytest
import torch

from vmas_pusht.physical_abstraction import physical_state, setup_scene
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.snapshot import capture, restore
from vmas_pusht.stable_composition import run_group
from vmas_pusht.recovery_attribution import (
    PhysicsObserver, reconstruct_retreats, run_factorial, endpoint_metrics,
    compare_resolutions, summarize_analysis,
)


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def tiny_source():
    group=run_group(profile='dry_light',substeps=5,supports=['passive','world_anchor','preloaded_anchor'],
        planners=['analytic_feedback'],angles_deg=[0.],nuisances=[(0.,0.,0.)],
        rotation_steps=5,retreat_steps=2,max_recovery=6,model_training_profile='test')
    return dict(groups=[group])


def test_observer_preserves_bitwise_trajectory_and_records_actual_work():
    env=make_profile('dry_light',num_envs=2,substeps=5);setup_scene(env)
    snapshot=capture(env)
    command=[torch.tensor([[0.,-.5],[0.,0.]]),torch.tensor([[-.2,.2],[0.,0.]])]
    original_method=env.world.update_env_forces
    env.step(command);expected=physical_state(env).clone();restore(env,snapshot)
    old_positions=torch.stack([a.state.pos.clone() for a in env.world.agents],1)
    with PhysicsObserver(env) as observer:
        observer.begin_step();env.step(command);measure=observer.end_step()
    assert torch.equal(expected,physical_state(env))
    assert env.world.update_env_forces==original_method
    new_positions=torch.stack([a.state.pos for a in env.world.agents],1)
    force=torch.stack(command,1)
    work=(force*(new_positions-old_positions)).sum(-1)
    torch.testing.assert_close(measure['actuator_signed_work'],work,atol=1e-8,rtol=1e-5)
    assert measure['actuator_absolute_work'].ge(measure['actuator_signed_work'].abs()-1e-8).all()
    assert measure['actuator_absolute_work'][1].eq(0).all()
    assert measure['contact_force_peak_norm'][1].gt(0).any(), 'Zero drive does not remove collision forces'
    assert measure['contact_force_peak_substep'].max()<5
    assert measure['contact_force_impulse'].shape==(2,2,2,2)


def test_observer_lifecycle_rejects_incomplete_steps_and_restores_after_error():
    env=make_profile('dry_light',num_envs=1,substeps=5)
    original=env.world._integrate_state
    with pytest.raises(RuntimeError):
        with PhysicsObserver(env) as observer:
            observer.begin_step();observer.begin_step()
    assert env.world._integrate_state==original


def test_replay_matches_all_original_times_before_extracting(tiny_source):
    prepared=reconstruct_retreats(tiny_source,source_substeps=5)
    assert len(prepared['cases'])==3
    assert all(check['bitwise_all_times'] for check in prepared['replay_checks'])
    assert all(check['maximum_absolute_state_error']==0 for check in prepared['replay_checks'])
    assert [case['source_row'] for case in prepared['cases']]==[0,1,2]
    assert [case['support'] for case in prepared['cases']]==['passive','world_anchor','preloaded_anchor']


def test_replay_detects_corrupt_source_actions(tiny_source):
    corrupted=copy.deepcopy(tiny_source)
    corrupted['groups'][0]['actions'][0,0,0]=torch.tensor([.5,.5])
    with pytest.raises(AssertionError,match='not bitwise exact'):
        reconstruct_retreats(corrupted,source_substeps=5)


def test_factorial_off_is_zero_drive_but_keeps_physical_bodies(tiny_source):
    prepared=reconstruct_retreats(tiny_source,source_substeps=5)
    run=run_factorial(prepared,substeps=5,horizon=6)
    assert run['trajectory'].shape==(12,7,24)
    starts=run['trajectory'][:,0].reshape(3,4,24)
    assert torch.equal(starts,starts[:,:1].expand_as(starts))
    assert run['actions'].norm(dim=-1).max()<=1.000001
    for i,row in enumerate(run['rows']):
        for agent,flag in enumerate((row['main_on'],row['helper_on'])):
            if not flag:
                assert run['actions'][i,:,agent].eq(0).all()
                assert run['actual_actuator_forces'][i,:,agent].eq(0).all()
                assert run['measurements']['actuator_absolute_work'][i,:,agent].eq(0).all()
    assert run['finite'].all()
    analysis=endpoint_metrics(run,prepared)
    assert len(analysis['endpoints'])==24
    assert len(analysis['contrasts'])==6
    assert len(analysis['historical_D_checks'])==3
    assert len(summarize_analysis(analysis)['endpoint_summary'])==24
    assert all('first_goal_passed' in row for row in analysis['endpoints'])


def test_observer_does_not_change_feedback_branching(tiny_source):
    prepared=reconstruct_retreats(tiny_source,source_substeps=5)
    observed=run_factorial(prepared,substeps=5,horizon=6,instrument=True)
    plain=run_factorial(prepared,substeps=5,horizon=6,instrument=False)
    for key in ('trajectory','actions','controller_phase','ready','helper_anchor'):
        assert torch.equal(observed[key],plain[key]),key


def test_refinement_uses_same_full_start_not_different_closed_loop_sources(tiny_source):
    prepared=reconstruct_retreats(tiny_source,source_substeps=5)
    a=run_factorial(prepared,substeps=5,horizon=6)
    b=run_factorial(prepared,substeps=10,horizon=6)
    assert torch.equal(a['trajectory'][:,0],b['trajectory'][:,0])
    comparisons=compare_resolutions([a,b],prepared)
    assert len(comparisons)==12
    assert all(row['identical_complete_initial_state'] for row in comparisons)
    b['trajectory'][0,0,18]+=.0001
    with pytest.raises(AssertionError,match='changed full initial'):
        compare_resolutions([a,b],prepared)


def test_factorial_contrasts_include_nonadditive_interaction(tiny_source):
    prepared=reconstruct_retreats(tiny_source,source_substeps=5)
    run=run_factorial(prepared,substeps=5,horizon=6)
    # Synthetic endpoint responses on a real, fully populated result isolate
    # the estimand: 1,3,4,10 imply main2, helper3, interaction4, not D-A=9.
    for branch,theta in enumerate((1.,3.,4.,10.)):
        run['trajectory'][branch,-1,16]=run['trajectory'][branch,0,16]+theta
    analysis=endpoint_metrics(run,prepared)
    row=next(row for row in analysis['contrasts'] if row['case_id']==0 and row['endpoint']=='fixed_6_steps')
    assert row['delta_angle_rad_B_minus_A']==pytest.approx(2.)
    assert row['delta_angle_rad_C_minus_A']==pytest.approx(3.)
    assert row['delta_angle_rad_D_minus_B_minus_C_plus_A']==pytest.approx(4.)
