from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('mujoco')

from uav_transport import mission
from uav_transport.control import RecoveryConfig,TransportController
from uav_transport.local_model import FEATURE_NAMES
from uav_transport.plant import TransportConfig,TransportEnv
from uav_transport.snapshot import capture,physical_state,restore


class FakePredictor:
    def __init__(self):self.inputs=[]
    def predict(self,x):
        self.inputs.append(x.copy());n=len(x)
        output=np.tile(np.array([.2,1.,10.,.1,1.,.01,.01,1.]),(n,1))
        mode=x[:,FEATURE_NAMES.index('operation_reconfigure')]>0
        output[mode,0]=.02;output[mode,1]=.3
        output[:,2]=x[:,FEATURE_NAMES.index('operation_rearrange_seconds')]+x[:,FEATURE_NAMES.index('operation_retension_seconds')]+5.
        return output


def fixture():
    cfg=TransportConfig();env=TransportEnv(cfg);controller=TransportController(cfg)
    return env,controller


def test_case_matrix_has_twelve_distinct_conditions_and_two_new_goal_compositions():
    cases=mission.mission_cases()
    assert len(cases)==len({x['case_id'] for x in cases})==12
    assert {x['payload_mass'] for x in cases}=={.026,.034}
    assert {x['donor'] for x in cases}=={0,1,2}
    a=np.asarray(cases[0]['waypoint_offsets']);b=np.asarray(cases[1]['waypoint_offsets'])
    np.testing.assert_allclose(b[:,:2],a[:,:2]@np.array([[0.,1.],[-1.,0.]]),atol=0)
    assert cases[0]['slack_distance']==.07 and cases[1]['slack_distance']==.09


def test_selector_applies_identical_constraint_criterion_and_explicit_minimum_risk_fallback():
    predicted=np.array([[.02,.3,9.,.2,1.,.01,.01,1.],[.01,.3,8.,.2,1.,.01,.01,1.],
                        [.2,.9,6.,.2,1.,.01,.01,1.]])
    scores,feasible,fallback=mission.outcome_scores(predicted)
    assert feasible.tolist()==[True,True,False] and not fallback and scores.argmin()==1
    scores,feasible,fallback=mission.outcome_scores(predicted,margins=(.06,.4))
    assert fallback and not feasible.any() and scores.argmin()==1
    assert np.isfinite(scores).all()


def test_learned_selection_queries_only_frozen_features_and_keeps_parent_unchanged(monkeypatch):
    env,controller=fixture();predictor=FakePredictor()
    def forbidden(*args,**kwargs):raise AssertionError('Learned policy queried future physics')
    monkeypatch.setattr(mission,'fork',forbidden);monkeypatch.setattr(env,'step',forbidden)
    before=capture(env);memory=controller.capture_memory()
    mode,config,record=mission.select_option(env,controller,0,.07,'wm_small',predictor,margins=(.01,.1))
    assert mode and record['candidate_count']==18 and record['predictor_evaluations']==18
    assert record['simulated_candidate_control_steps']==0
    assert record['feasible_count']==9 and not record['minimum_risk_fallback']
    assert config.rearrange_seconds==1. and config.retension_seconds==1.5
    np.testing.assert_array_equal(before.state,capture(env).state)
    assert mission._memory_equal(memory,controller.capture_memory())
    mode,_,record=mission.select_option(env,controller,0,.07,'wm_small_direct',predictor,margins=(.01,.1))
    assert not mode and record['candidate_count']==9 and record['minimum_risk_fallback']


def test_exact_selector_forks_each_candidate_without_stepping_the_parent(monkeypatch):
    env,controller=fixture();before=capture(env);called=[]
    bank=[(False,RecoveryConfig()),(True,RecoveryConfig())]
    monkeypatch.setattr(mission,'candidate_bank',lambda *args,**kwargs:bank)
    def fake_operation(child,control,config,donor,mode,**kwargs):
        assert child is not env and control is not controller
        called.append(mode)
        child.step(np.zeros((3,4)))
        return dict(label=np.array([.02 if mode else .2,.3,9.,.1,1.,.01,.01,1.]),
                    summary=dict(control_steps=1))
    monkeypatch.setattr(mission,'run_option',fake_operation)
    mode,_,record=mission.select_option(env,controller,0,.07,'true_option')
    assert mode and called==[False,True]
    assert record['simulated_candidate_control_steps']==2
    np.testing.assert_array_equal(before.state,capture(env).state)


def test_rules_use_complete_physical_feedback_without_prediction_calls():
    env,controller=fixture()
    for planner,expected in [('rule_fixed',False),('rule_reconfigure',True)]:
        mode,config,record=mission.select_option(env,controller,1,.09,planner)
        assert mode==expected and config.slack_distance==.09
        assert config.rearrange_seconds==config.retension_seconds==2.
        assert record['predictor_evaluations']==record['simulated_candidate_control_steps']==0


def test_finite_phase_budgets_retain_failed_legs_and_execute_later_stages():
    config=mission.MissionConfig(settle_seconds=.04,move_seconds=.04,leg_budget_seconds=.1,
        goal_dwell_seconds=.02,option_budget_seconds=.1)
    result=mission.run_mission(mission.mission_cases()[0],'rule_reconfigure',config=config)
    summary=result['summary']
    assert not summary['delivery_completion'] and not summary['constrained_completion']
    assert len(result['stages'])==6 and len(result['operations'])==2
    assert result['stages'][-1]['name']=='leg_3'
    assert summary['control_steps']==27 and summary['total_seconds']==pytest.approx(.54)
    assert result['trace']['actions'].shape==(27,3,4)
    assert result['trace']['physical_states'].shape==(28,64)
    assert len(result['trace']['substeps']['time'])==270


def test_full_mission_carries_failed_prefix_and_replays_every_real_action(monkeypatch):
    original=mission._flight_stage
    def with_failed_first(*args,**kwargs):
        result=original(*args,**kwargs)
        if kwargs['name']=='leg_1':result[0]['success']=False
        return result
    monkeypatch.setattr(mission,'_flight_stage',with_failed_first)
    result=mission.run_mission(mission.mission_cases()[0],'rule_reconfigure')
    row=result['summary']
    assert row['waypoint_successes']==[False,True,True]
    assert row['option_completions']==row['actual_slack_successes']==[True,True]
    assert not row['delivery_completion']
    assert row['finite'] and not row['collision'] and row['warning_count']==0
    assert max(row['option_peak_excursions'])<.05 and max(row['option_peak_donor_tensions'])<.6
    env=TransportEnv(TransportConfig(**result['plant_config']));restore(env,result['initial_snapshot'])
    np.testing.assert_array_equal(physical_state(env),result['trace']['physical_states'][0])
    for step,(command,external) in enumerate(zip(result['trace']['actions'],result['trace']['external_forces'])):
        env.step(command,external)
        np.testing.assert_array_equal(physical_state(env),result['trace']['physical_states'][step+1])
    assert env.control_steps==row['control_steps']
    assert row['maximum_rotor_force']<=env.config.rotor_max


def test_tighter_termination_guard_executes_and_charges_extra_physical_steps():
    def execute(config):
        env=TransportEnv(TransportConfig(payload_mass=.026,formation_yaw=np.pi/6))
        controller=TransportController(env.config);origin=env.observe()['payload_pos'].copy()
        mission._flight_stage(env,controller,origin,name='settle',config=config,settle=True)
        return mission._flight_stage(env,controller,origin+np.array([.2,.1,.05]),name='leg_1',config=config)
    old_config=mission.MissionConfig(termination_position_tolerance=.025,termination_speed_tolerance=.05)
    old,old_block,_=execute(old_config)
    guarded,guarded_block,_=execute(mission.MissionConfig())
    assert old['success'] and guarded['success'] and guarded['guard_confirmed']
    assert guarded['end_step']>old['end_step']
    assert guarded['elapsed_seconds']>old['elapsed_seconds']
    assert guarded['maximum_seconds']==old['maximum_seconds']==6.
    assert guarded['goal_dwell_steps']>=guarded['termination_dwell_steps']>=guarded['required_dwell_steps']
    assert guarded['elapsed_seconds']==pytest.approx(len(guarded_block['actions'])*.02)
    np.testing.assert_array_equal(guarded_block['physical_states'][:len(old_block['physical_states'])],old_block['physical_states'])
    np.testing.assert_array_equal(guarded_block['actions'][:len(old_block['actions'])],old_block['actions'])


def test_guard_timeout_preserves_a_fulfilled_external_goal_at_the_original_budget():
    env,controller=fixture();goal=env.observe()['payload_pos'].copy()
    config=mission.MissionConfig(termination_position_tolerance=1e-8,termination_speed_tolerance=1e-8)
    stage,block,_=mission._flight_stage(env,controller,goal,name='leg_1',config=config)
    assert stage['elapsed_seconds']==6. and len(block['actions'])==300
    assert stage['success'] and not stage['guard_confirmed']
    assert stage['termination_dwell_steps']==0
    assert stage['goal_dwell_steps']>=stage['required_dwell_steps']
    for obs in block['observations'][-stage['required_dwell_steps']:]:
        assert np.linalg.norm(obs['payload_pos']-goal)<=config.position_tolerance
        assert np.linalg.norm(obs['payload_vel'])<=config.speed_tolerance


def test_settle_retains_fixed_duration_and_external_scoring_when_guard_is_unmet():
    env,controller=fixture();goal=env.observe()['payload_pos'].copy()
    config=mission.MissionConfig(termination_position_tolerance=1e-8,termination_speed_tolerance=1e-8)
    stage,block,_=mission._flight_stage(env,controller,goal,name='settle',config=config,settle=True)
    assert stage['elapsed_seconds']==2. and len(block['actions'])==100
    assert stage['success'] and not stage['guard_confirmed']
    assert not stage['termination_guard_applied']


@pytest.mark.parametrize('overrides',[
    {'termination_position_tolerance':0.},
    {'termination_speed_tolerance':-.01},
    {'termination_position_tolerance':.026},
    {'termination_speed_tolerance':.051},
])
def test_internal_termination_bounds_are_positive_and_within_external_bounds(overrides):
    with pytest.raises(ValueError,match='termination bounds'):
        mission.MissionConfig(**overrides)
