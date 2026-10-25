from functools import partial

import pytest
import torch

from vmas_pusht import interface_factorial as experiment
from vmas_pusht import interface_factorial_planner as factorial
from vmas_pusht import option_planner as original
from vmas_pusht.action_interface import candidate_table, ActionInterface
from vmas_pusht.interface_composition import run_group
from vmas_pusht.physical_abstraction import setup_scene, coarse_state, physical_state
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.stable_composition import empty_history


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


def models(seen=None):
    def predict(coarse, full=None, history=None):
        if seen is not None:
            seen.append(coarse.clone())
        result = torch.zeros(len(coarse), 5)
        result[:, 2] = -.06*coarse[:, -2]-.07*coarse[:, -1]
        result[:, 3] = coarse[:, 14]+2.*(coarse[:, -2]+coarse[:, -1])
        result[:, 4] = 1.
        return result
    keys = [f"passive/{model}/seed0" for model in ("full_matched", "full_large")]
    return dict(checkpoints={key:dict(reference_substeps=160,
        elapsed_time_q95_absolute_residual_seconds=.1) for key in keys},
        predictors={key:predict for key in keys})


def arguments(n=2):
    env, interface, history, initial = fixture(n)
    rows = [dict(planner="wm_full",support="passive",model_seed=0) for _ in range(n)]
    return (env,interface,rows,initial,torch.full((n,),.04),torch.full((n,),60),history,
            torch.ones(n,dtype=torch.bool)), dict(model_pack=models(),allow_routes=torch.ones(n,dtype=torch.bool))


def test_allocation_preserves_program_marginals_and_covers_every_route_without_duplicates():
    direct = candidate_table(expanded=False)
    allocated = [factorial.allocated_table(s) for s in range(3)]
    for table in allocated:
        torch.testing.assert_close(table[:,1:],direct[:,1:],atol=0,rtol=0)
        assert len(torch.unique(table,dim=0)) == 39
        assert torch.bincount(table[:,0].long()).tolist() == [15,12,12]
        assert table[table[:,2] == 0,0].eq(0).all()
    for i in (direct[:,2]>0).nonzero().flatten():
        assert sorted(float(t[i,0]) for t in allocated) == [0.,1.,2.]


def test_allocated_interface_uses_direct_bank_before_final_and_after_acquisition():
    env,interface,_,_=fixture(3)
    interface.state.mode[2]=0; interface.state.phase[2]=3
    tables,_=factorial.option_tables(env,interface,torch.tensor([False,True,True]),"budget39_s1")
    torch.testing.assert_close(tables[0],candidate_table(expanded=False),atol=0,rtol=0)
    torch.testing.assert_close(tables[1],factorial.allocated_table(1),atol=0,rtol=0)
    torch.testing.assert_close(tables[2],tables[0],atol=0,rtol=0)


def test_unrestricted_learned_planner_matches_historical_selection_and_predictions():
    args,kwargs=arguments()
    for policy in ("direct","expanded"):
        allowed=torch.tensor([False,True]) if policy=="expanded" else torch.zeros(2,dtype=torch.bool)
        kwargs["allow_routes"]=allowed
        a,da=original.select_options(*args,**kwargs)
        b,db=factorial.select_options(*args,**kwargs,policy=policy)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for key in ("selected_prediction","selected_score","chosen_index","predicted_steps","candidate_count"):
            torch.testing.assert_close(da[key],db[key],atol=0,rtol=0,equal_nan=True)


def test_budget_queries_are_actual_frozen_model_evaluations_and_goals_do_not_enter_inputs(monkeypatch):
    args,kwargs=arguments(); seen=[]
    kwargs["model_pack"]=models(seen)
    def forbidden(*args,**kwargs): raise AssertionError("Learned policy queried a future simulator")
    monkeypatch.setattr(factorial,"true_option_predictions",forbidden)
    for goal in (.04,.2):
        values=list(args);values[4]=torch.full((2,),goal)
        _,diag=factorial.select_options(*values,**kwargs,policy="budget39_s0")
        assert diag["candidate_budget"].tolist()==diag["evaluated_candidates"].tolist()==[39,39]
        assert diag["simulated_candidate_steps"].eq(0).all()
    assert len(seen)==2 and len(seen[0])==78
    torch.testing.assert_close(seen[0],seen[1],atol=0,rtol=0)


def test_true_direct_uses_original_options_and_leaves_live_state_unchanged():
    args,kwargs=arguments(1);args[2][0]["planner"]="true_option";kwargs["model_pack"]=None
    before=physical_state(args[0]).clone()
    a,da=original.select_options(*args,**dict(kwargs,allow_routes=torch.zeros(1,dtype=torch.bool)))
    b,db=factorial.select_options(*args,**kwargs,policy="direct")
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(da["selected_prediction"],db["selected_prediction"],atol=0,rtol=0)
    torch.testing.assert_close(before,physical_state(args[0]),atol=0,rtol=0)
    assert db["evaluated_candidates"].tolist()==[39]
    assert db["simulated_candidate_steps"].tolist()==[13*(5+10+20)]
    assert db["padded_simulator_steps"].tolist()==[39*20]


def test_oracle_grouped_tables_keep_row_alignment_and_only_count_enabled_rollouts(monkeypatch):
    args,kwargs=arguments(2)
    for row in args[2]:row["planner"]="true_option"
    kwargs["model_pack"]=None
    args[1].state.mode[1]=0
    args[5][1]=5
    calls=[]
    def oracle(env,interface,table,indices,legal):
        calls.append((indices.clone(),table.clone()))
        out=torch.zeros(len(indices),len(table),5)
        out[:,:,2]=table[:,0]*-.01
        out[:,:,3]=table[:,3]*.1
        out[:,:,4]=legal[indices].float()
        return out
    monkeypatch.setattr(factorial,"true_option_predictions",oracle)
    _,diag=factorial.select_options(*args,**kwargs,policy="budget39_s0")
    assert len(calls)==2
    assert diag["evaluated_candidates"].tolist()==[39,13]
    assert diag["simulated_candidate_steps"].tolist()==[455,65]
    assert diag["padded_simulator_steps"].tolist()==[780,195]
    assert diag["oracle_calls"]==2
    assert diag["candidate_table"][1,:,0].eq(0).all()


def test_direct_deadline_uses_known_duration_without_masking_safe_zero():
    args,kwargs=arguments(1);args[5][0]=5
    _,diag=factorial.select_options(*args,**kwargs,policy="direct")
    assert not diag["budget_fallback"].any()
    assert diag["selected_elapsed_margin_seconds"].eq(0).all()
    args[5][0]=4
    option,diag=factorial.select_options(*args,**kwargs,policy="direct")
    assert diag["budget_fallback"].all() and option.tolist()==[[0.,0.,0.,5.]]


def test_factorial_schedule_is_complete_without_repeating_true_controls_over_fit_seeds():
    schedule=experiment.jobs()
    assert len(schedule)==70 and len(set(schedule))==70
    assert sum(p=="true_option" for p,_,_,_ in schedule)==10
    assert len({experiment.arm_name(p,policy) for p,policy,_,_ in schedule})==15
    with pytest.raises(ValueError,match="unique"):
        experiment.jobs(model_seeds=(0,0))


def test_full_runner_override_restricts_final_routes_and_preserves_task_clock():
    calls=[]
    def zero(env,interface,rows,initial,goals,remaining,history,query_mask,**kwargs):
        calls.append(kwargs["allow_routes"].clone())
        option=torch.tensor([0.,0.,0.,5.]).expand(env.num_envs,4).clone()
        return option,{}
    group=run_group(substeps=2,planners=["true_option"],supports=["passive"],
        angles_deg=[0.],nuisances=[(0.,0.,0.)],rotation_steps=5,retreat_steps=2,max_recovery=6,
        observe=False,option_selector=zero,expanded_final_interface=False)
    assert calls and not any(x.any() for x in calls)
    assert not group["rows"][0]["expanded_final_interface"]
    result=experiment.relabel(group,"true_option","direct")
    assert result["rows"][0]["planner"]=="true_option_direct"
    assert result["rows"][0]["base_planner"]=="true_option"
    assert result["rows"][0]["total_steps"]==result["rows"][0]["task_steps"]+10
    assert all(r["planner"]=="true_option_direct" for r in result["option_records"])
