"""Exploratory angular-state diagnostic keeps the primary experiment frozen."""
import copy

import pytest
import torch

from vmas_pusht.abstraction_models import save_artifact
from vmas_pusht.angular_ablation import (
    BAR_OMEGA_FULL_INDEX, fit_angular, omega_conditioned_alias, omega_features,
    predict_entry, primary_strict_sidecar, registered_protocol,
)
from vmas_pusht.stable_local_wm import collect_local


@pytest.fixture(scope="module")
def small_data():
    torch.set_num_threads(1)
    contexts = [dict(context_id=i, angle_deg=0., primary_x_offset=0.,
        velocity_omega=[0., 0., .03*i], context_split="train") for i in range(2)]
    programs = [dict(program_id=i, direction_deg=d, strength=s, duration_steps=t, action_split=split)
        for i, (d,s,t,split) in enumerate([(0.,.3,10,"train"), (10.,.45,10,"val"), (35.,.75,15,"test_action")])]
    return collect_local("original", substeps=2, contexts=contexts, programs=programs,
                         supports=("passive", "world_anchor"))


def test_angular_state_is_exactly_one_measured_feature_and_fixed_resolution(small_data):
    protocol = registered_protocol()
    assert protocol["parameters"] == 1708 <= 1763
    assert protocol["additional_state_scalars"] == 1
    assert protocol["action_selection"] is False
    coarse, full = small_data["X_coarse"][:3], small_data["X_full"][:3]
    x = omega_features(coarse, full, 160)
    assert x.shape == (3,19)
    torch.testing.assert_close(x[:,18], full[:,26])
    changed = full.clone()
    changed[:,26] += .123
    shifted = omega_features(coarse,changed,160)
    torch.testing.assert_close(shifted[:,:18], x[:,:18], atol=0, rtol=0)
    torch.testing.assert_close(shifted[:,18]-x[:,18], torch.full((3,),.123))
    refined_c, refined_f = coarse.clone(), full.clone()
    refined_c[:,15]=320; refined_f[:,15]=320
    torch.testing.assert_close(omega_features(refined_c,refined_f,160),x,atol=0,rtol=0)


def test_prediction_budget_train_only_normalization_and_safe_checkpoint(small_data,tmp_path):
    data = small_data
    report, artifact = fit_angular(data,steps=3,seeds=(1,))
    changed = copy.deepcopy(data)
    for i,row in enumerate(changed["rows"]):
        if row["split"] != "train":
            changed["Y"][i] += 10
    _, second = fit_angular(changed,steps=3,seeds=(1,))
    for key, entry in artifact["checkpoints"].items():
        assert entry["parameter_count"] == 1708
        assert all(data["rows"][i]["split"] == "train" for i in entry["train_row_indices"])
        for name, value in entry["state_dict"].items():
            torch.testing.assert_close(value,second["checkpoints"][key]["state_dict"][name],atol=0,rtol=0)
        torch.testing.assert_close(artifact["predictions"][key]["prediction"],
                                  second["predictions"][key]["prediction"],atol=0,rtol=0)
        indices=artifact["predictions"][key]["row_indices"]
        prediction=predict_entry(entry,data["X_coarse"][indices],data["X_full"][indices])
        torch.testing.assert_close(prediction,artifact["predictions"][key]["prediction"],atol=0,rtol=0)
    path=tmp_path/"angular.pt"
    save_artifact(path,{"stage":"exploratory"},artifact)
    assert set(torch.load(path)["checkpoints"])==set(artifact["checkpoints"])
    assert {r["split"] for r in report["summary"]}=={"test_action","strict_test_only"}


def test_strict_test_sidecar_excludes_validation_without_changing_predictions(small_data):
    _, artifact=fit_angular(small_data,steps=2,seeds=(0,))
    before={k:v["prediction"].clone() for k,v in artifact["predictions"].items()}
    report=primary_strict_sidecar(small_data,artifact)
    assert report["validation_in_strict_test"] is False
    for row in report["seed_metrics"]:
        expected=sum(r["mode"]==row["mode"] and r["split"]=="test_action" for r in small_data["rows"])
        assert row["rows"]==expected
    changed=copy.deepcopy(small_data)
    for i,row in enumerate(changed["rows"]):
        if row["split"]=="val":
            changed["Y"][i]+=1000
    assert primary_strict_sidecar(changed,artifact)["seed_metrics"]==report["seed_metrics"]
    for key,values in artifact["predictions"].items():
        torch.testing.assert_close(values["prediction"],before[key],atol=0,rtol=0)


def test_conditioned_alias_reports_coverage_and_like_for_like_subset():
    coarse=torch.zeros(4,18)
    full=torch.zeros(4,32)
    full[:,26]=torch.tensor([0.,0.,1.,-1.])
    physical=torch.zeros(4,24); physical[:,17]=full[:,26]
    y=torch.tensor([[0.,0.,0.],[2.,0.,2.],[100.,0.,100.],[-100.,0.,-100.]])
    data=dict(rows=[dict(mode="passive",angle_deg=0.) for _ in range(4)],
              X_coarse=coarse,X_full=full,Y=y,physical_initial=physical,profile={"name":"synthetic"},substeps=2)
    summary=omega_conditioned_alias(data)["summary"][0]
    assert summary["repeated_omega_conditioned_rows"]==2
    assert summary["repeated_omega_conditioned_fraction"]==.5
    assert summary["repeated_group_omega_values_rad_per_time"]==[0.]
    assert summary["geometry_plus_omega"]["conditional_rmse_xy_units_rad"]==[1.,0.,1.]
    assert summary["geometry_on_same_repeated_omega_subset"]["conditional_rmse_xy_units_rad"]==[1.,0.,1.]
    assert summary["geometry_all_rows"]["conditional_rmse_xy_units_rad"][2]>60
    # Omitting the two difficult singleton groups is not hidden as a helper effect.
    data["physical_initial"][0,17]=5
    with pytest.raises(ValueError,match="disagrees"):
        omega_conditioned_alias(data)


def test_full_sensor_is_required_not_replaced_with_unknown_zero(small_data):
    with pytest.raises(ValueError,match="requires full"):
        omega_features(small_data["X_coarse"][:2],None,160)
