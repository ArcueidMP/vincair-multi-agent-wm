"""Common-state oracle labels, genuine history, and read-only replay checks."""
import copy
import math

import pytest
import torch

from vmas_pusht import interface_model_audit as audit
from vmas_pusht.physical_abstraction import coarse_state, helper_action, physical_state, setup_scene
from vmas_pusht.physics_profiles import make_profile
from vmas_pusht.scenario import wrap
from vmas_pusht.stable_composition import history_inputs, snapshot_record


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source_group():
    env = make_profile("dry_light", num_envs=2, substeps=2, seed=101)
    anchor = setup_scene(env, angle=[0., .2], hidden_v=[[.02, 0.], [0., .02]], hidden_omega=[.05, -.05])
    rows = [dict(planner="true_option", support=s, initial_angle_deg=angle, nuisance_id=i,
                 first_goal_rad=.04, relative_goal_rad=.08, substeps=2)
            for i,(s,angle) in enumerate(zip(("passive", "world_anchor"),(0., 11.459)))]
    def helper():
        return helper_action(env,"world_anchor",anchor)*torch.tensor([[0.],[1.]])
    geometry, previous, prelude = [coarse_state(env).clone()], [], [physical_state(env).clone()]
    for _ in range(10):
        a = torch.stack((torch.zeros(2,2),helper()),1)
        env.step([a[:,0],a[:,1]])
        previous.append(a.clone());geometry.append(coarse_state(env).clone());prelude.append(physical_state(env).clone())
    initial = env.scenario.t_pose().clone()
    start = snapshot_record(env,[0,1],anchor,history_inputs(geometry,previous),rows,"post_prelude")
    start["interface_memory"] = dict(anchor=anchor.clone(),primitive=dict(mode=torch.full((2,),3)))
    states,actions,memories = [physical_state(env).clone()],[],[]
    start_poses={0:env.scenario.t_pose().clone()}
    main=torch.zeros(2,2)
    angle=env.scenario.bar.state.rot[:,0]
    main[0]=torch.stack((angle[0].sin(),-angle[0].cos()))*.3
    for step in range(10):
        if step==5:
            start_poses[5]=env.scenario.t_pose().clone()
            main.zero_();angle=env.scenario.bar.state.rot[:,0]
            main[1]=torch.stack((angle[1].sin(),-angle[1].cos()))*.3
        a=torch.stack((main,helper()),1)
        env.step([a[:,0],a[:,1]])
        actions.append(a.clone());states.append(physical_state(env).clone())
        memories.append(dict(interface=dict(anchor=anchor.clone(),primitive=dict(mode=torch.full((2,),3)))))
    trajectory=torch.stack(states,1)
    labels=[]
    for row,step in ((0,0),(1,5)):
        pose=torch.cat((trajectory[row,step+5,12:14],trajectory[row,step+5,16:17]))
        delta=pose-start_poses[step][row];delta[2]=wrap(delta[2])
        labels.append(torch.cat((delta,torch.tensor([.5,1.]))))
    records=[]
    for j,(row,step) in enumerate(((0,0),(1,5))):
        records.append(dict(row=row,planner="true_option",task_phase="rotate_first",start_step=step,
            start_env_step=10.+step,start_pose=start_poses[step][row].tolist(),start_contact_mode=3,
            option=[0.,0.,.3,5.],predicted_outcome=labels[j].tolist(),chosen_index=6,
            interrupted=bool(j),completed=not bool(j),elapsed_steps=5 if not j else 2,
            elapsed_seconds=.5 if not j else .2,termination_reason="option_finished" if not j else "stage_goal_dwell",
            actual_response=labels[j][:3].tolist() if not j else [99.,99.,99.]))
    records.append(dict(records[0],row=1,start_pose=start_poses[0][1].tolist(),option=[0.,0.,0.,5.],
                        predicted_outcome=None,chosen_index=-1,interrupted=False))
    queries=[]
    for step in (0,5):
        opts=torch.zeros(2,4);opts[:,3]=5
        pred=torch.full((2,5),float("nan"));mask=torch.zeros(2,dtype=torch.bool)
        for r in records:
            if r["start_step"]==step:
                i=r["row"];mask[i]=True;opts[i]=torch.tensor(r["option"])
                if r["predicted_outcome"] is not None:pred[i]=torch.tensor(r["predicted_outcome"])
        queries.append(dict(step=step,query_mask=mask,options=opts,
            diagnostics=dict(source=["true_simulator"]*2,selected_prediction=pred)))
    return dict(rows=rows,physics_profile=env.physics_profile,trajectory=trajectory,
        actions=torch.stack(actions,1),prelude_trajectory=torch.stack(prelude,1),
        prelude_actions=torch.stack(previous,1),initial_pose=initial,
        helper_anchor=anchor[:,None].expand(2,11,2).clone(),counterfactual_snapshots=[start],
        controller_memory=memories,option_records=records,queries=queries)


def test_physical_geometry_mapping_matches_live_geometry_exactly():
    env=make_profile("dry_light",num_envs=3,substeps=2)
    setup_scene(env,angle=[-.2,.0,.52],hidden_omega=[-.1,.2,.0])
    torch.testing.assert_close(audit.geometry_from_physical(physical_state(env)),coarse_state(env),atol=0,rtol=0)


def test_oracle_replay_preserves_true_full_labels_when_real_option_was_interrupted():
    group=source_group()
    data=audit.extract_oracle_queries(group,source_id="synthetic")
    assert data["replay"]["bitwise_all_times"]
    assert data["replay"]["replay_steps"]==10
    assert len(data["rows"])==2 and len(data["excluded"])==1
    assert data["rows"][1]["real_interrupted"]
    torch.testing.assert_close(data["truth"][1],torch.tensor(group["option_records"][1]["predicted_outcome"]),atol=0,rtol=0)
    assert data["truth"][1,:3].abs().max()<1
    assert data["truth"][1,3]==.5
    assert data["history"][:,:,16].eq(1).all()
    assert len(data["complete_label_vs_actual_errors"])==1
    assert data["coarse"].shape==(2,24) and data["full"].shape==(2,38)


def test_wm_guesses_cannot_masquerade_as_true_simulator_labels():
    group=source_group()
    group["queries"][0]["diagnostics"]["source"][0]="world_model"
    with pytest.raises(ValueError,match="not a simulator"):
        audit.extract_oracle_queries(group)


def test_corrupt_actions_or_histories_fail_replay_validation():
    group=source_group(); changed=copy.deepcopy(group)
    changed["actions"][0,2,0,0]+=.2
    with pytest.raises(AssertionError,match="replay diverged"):
        audit.extract_oracle_queries(changed)
    changed=copy.deepcopy(group)
    changed["prelude_actions"][0,1,0,0]+=.2
    with pytest.raises(AssertionError,match="history differs"):
        audit.extract_oracle_queries(changed)


def test_query_weighted_and_episode_balanced_errors_are_explicit():
    data=dict(truth=torch.zeros(3,5),query_pose=torch.zeros(3,3),initial_pose=torch.zeros(3,3),
              goals=torch.zeros(3),rows=[dict(episode_id=x) for x in ("a","a","b")])
    prediction=data["truth"].clone();prediction[:2,2]=.1
    result=audit.score_predictions(data,prediction,[0,1,2],torch.ones(3),torch.ones(3,3))
    assert result["queries"]==3 and result["episodes"]==2
    assert result["angular_rmse_rad"]==pytest.approx(math.sqrt(.02/3),abs=1e-7)
    assert result["episode_balanced_angular_rmse_rad"]==pytest.approx(math.sqrt(.01/2),abs=1e-7)
