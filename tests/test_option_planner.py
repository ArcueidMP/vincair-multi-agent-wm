import pytest
import torch

from vmas_pusht import option_planner as planner
from vmas_pusht.action_interface import ActionInterface, valid_options
from vmas_pusht.physical_abstraction import setup_scene, physical_state, coarse_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.stable_composition import empty_history
from vmas_pusht.snapshot import capture


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(n=2):
    env = make_profile("dry_light", num_envs=n, substeps=2)
    anchor = setup_scene(env, angle=.1)
    interface = ActionInterface(env, ["passive"]*n, anchor)
    history = empty_history(coarse_state(env)); history[:, :, 16] = 1
    initial = env.scenario.t_pose().clone(); initial[:, 2] = 0
    return env, interface, history, initial


def pack(seen=None, margin=0.):
    def prediction(coarse, full=None, history=None):
        if seen is not None: seen.append((coarse.clone(), full.clone(), history.clone()))
        result = torch.zeros(len(coarse), 5)
        upper, lower = coarse[:, -2].bool(), coarse[:, -1].bool()
        result[upper, 2] = -.06
        result[lower, 2] = -.07
        result[:, 3] = coarse[:, 14]
        result[upper, 3] = 3.0
        result[lower, 3] = 3.7
        result[:, 4] = 1.
        return result
    key = "passive/full_matched/seed0"
    return dict(checkpoints={key:dict(reference_substeps=160,
        elapsed_time_q95_absolute_residual_seconds=margin)}, predictors={key:prediction})


def run_select(env, interface, history, initial, rows, **kwargs):
    args = dict(goals=torch.full((env.num_envs,), .04), remaining=torch.full((env.num_envs,), 60),
        history=history, query_mask=torch.ones(env.num_envs,dtype=torch.bool),
        model_pack=pack(), allow_routes=torch.ones(env.num_envs,dtype=torch.bool))
    args.update(kwargs)
    return planner.select_options(env,interface,rows,initial,**args)


def test_learned_route_selection_and_direct_ablation_use_same_model_without_simulator(monkeypatch):
    env,interface,history,initial=fixture()
    def forbidden(*args,**kwargs): raise AssertionError("Learned planner queried true future dynamics")
    monkeypatch.setattr(env,"step",forbidden)
    monkeypatch.setattr(planner,"true_option_predictions",forbidden)
    rows=[dict(planner=p,support="passive") for p in ("wm_full","wm_full_direct")]
    before=physical_state(env).clone()
    selected,diag=run_select(env,interface,history,initial,rows)
    assert selected[:,0].tolist()==[1.,0.]
    assert diag["source"]==["world_model","world_model"]
    torch.testing.assert_close(before,physical_state(env),atol=0,rtol=0)


def test_goals_change_costs_without_entering_model_inputs():
    env,interface,history,initial=fixture(1);seen=[]
    rows=[dict(planner="wm_full",support="passive")]
    for target in (.04,.2):
        run_select(env,interface,history,initial,rows,model_pack=pack(seen),goals=torch.tensor([target]))
    assert len(seen)==2
    for a,b in zip(seen[0],seen[1]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert seen[0][0][:,15].eq(160).all()


def test_route_time_and_validation_margin_cannot_be_ignored():
    env,interface,history,initial=fixture(1);rows=[dict(planner="wm_full",support="passive")]
    selected,diag=run_select(env,interface,history,initial,rows,remaining=torch.tensor([5]))
    assert selected[0,0]==0
    selected,diag=run_select(env,interface,history,initial,rows,remaining=torch.tensor([5]),model_pack=pack(margin=.1))
    assert not diag["budget_fallback"].any()
    assert diag["predicted_steps"].eq(5).all()
    assert diag["selected_elapsed_margin_seconds"].eq(0).all()
    selected,diag=run_select(env,interface,history,initial,rows,remaining=torch.tensor([4]),model_pack=pack(margin=.1))
    assert diag["budget_fallback"].all()
    assert selected[0].tolist()==[0.,0.,0.,5.]
    assert not torch.isfinite(diag["selected_prediction"]).any()


def test_known_direct_duration_does_not_hide_raw_model_timing_error():
    env,interface,history,initial=fixture(1);rows=[dict(planner="wm_full",support="passive")]
    models=pack(margin=1.)
    key="passive/full_matched/seed0";original=models["predictors"][key]
    def wrong_time(*args,**kwargs):
        result=original(*args,**kwargs);result[:,3]=2.;return result
    models["predictors"][key]=wrong_time
    _,diag=run_select(env,interface,history,initial,rows,remaining=torch.tensor([5]),model_pack=models)
    assert not diag["budget_fallback"].any()
    assert diag["selected_prediction"][0,3]==2.
    assert diag["predicted_steps"][0]==5.


def test_feedback_is_explicit_and_route_permission_is_enforced():
    env,interface,history,initial=fixture()
    rows=[dict(planner=p,support="passive") for p in ("feedback_upper","feedback_lower")]
    selected,diag=run_select(env,interface,history,initial,rows,model_pack=None)
    assert selected[:,0].tolist()==[1.,2.]
    selected,diag=run_select(env,interface,history,initial,rows,model_pack=None,allow_routes=torch.zeros(2,dtype=torch.bool))
    assert selected[:,0].eq(0).all()
    assert not diag["queried_mask"].any()


def test_true_option_oracle_restores_complete_states_without_touching_live_env():
    env,interface,_,_=fixture(1)
    before=capture(env)
    table=torch.tensor([[0.,0.,0.,5.],[0.,0.,.3,5.]])
    result=planner.true_option_predictions(env,interface,table,torch.tensor([0]),
        valid_options(env,interface.mode,table,state=interface.state))
    assert result.shape==(1,2,5)
    assert result[:,:,3].eq(.5).all()
    assert result[:,:,4].eq(1).all()
    after=capture(env)
    for a,b in zip(before.entities,after.entities):
        for key,value in a.state.items():
            if value is not None:torch.testing.assert_close(value,b.state[key],atol=0,rtol=0)
    torch.testing.assert_close(before.steps,after.steps,atol=0,rtol=0)
    assert not interface.busy.any()


def test_failed_navigation_does_not_advertise_an_acquired_face():
    env,interface,history,initial=fixture(1)
    interface.state.mode[:]=0;interface.state.phase[:]=1;interface.state.failed[:]=True
    rows=[dict(planner="feedback_upper",support="passive")]
    selected,_=run_select(env,interface,history,initial,rows,model_pack=None)
    assert selected[0,0]==0 and selected[0,2]==0


def test_missing_models_and_busy_options_fail_before_querying():
    env,interface,history,initial=fixture(1);rows=[dict(planner="wm_full",support="passive")]
    with pytest.raises(ValueError,match="Missing frozen"):
        run_select(env,interface,history,initial,rows,model_pack=None)
    interface.start(torch.tensor([[0.,0.,0.,5.]]))
    with pytest.raises(ValueError,match="committed option"):
        run_select(env,interface,history,initial,rows)
