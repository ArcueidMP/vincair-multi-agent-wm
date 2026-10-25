from copy import deepcopy
import itertools

import numpy as np
import pytest

pytest.importorskip("mujoco")

from uav_transport import selection_audit as audit
from uav_transport.local_model import FEATURE_NAMES, TARGET_NAMES


def rows_for_context(context_id=200, split="test", holdout=False):
    pairs = tuple(itertools.product((1.5, 2.5), (1.75, 2.5))) if holdout else audit.GRID
    return [dict(context_id=context_id, split=split, reconfigure=mode,
        recovery_config=dict(rearrange_seconds=a, retension_seconds=b),
        summary=dict(zip(TARGET_NAMES, [.02 if mode else .2, .3 if mode else 1.,
            4.+a+b, .1, .8, .005, .01, True])))
        for mode in (False, True) for a, b in pairs]


def synthetic_dataset():
    arrays, rows, contexts = {}, [], []
    for cid, split in ((0, "train"), (100, "val"), (200, "test"), (201, "test")):
        contexts.append(dict(context_id=cid, split=split))
        specs = [(split, False)]+([("test_action", True)] if split == "test" else [])
        for target_split, holdout in specs:
            for row in rows_for_context(cid, target_split, holdout):
                x = np.zeros(len(FEATURE_NAMES)); x[0] = cid+.1
                for feature, value in zip(audit.ACTION_FEATURES, (row["recovery_config"]["rearrange_seconds"],
                        row["recovery_config"]["retension_seconds"], float(row["reconfigure"]))):
                    x[FEATURE_NAMES.index(feature)] = value
                y = np.asarray([row["summary"][name] for name in TARGET_NAMES], dtype=float)
                for name, value in (("X", x), ("Y", y), ("groups", cid)):
                    arrays.setdefault(name+"_"+target_split, []).append(value)
                rows.append(row)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    metadata = dict(rows=rows, contexts=contexts, feature_names=list(FEATURE_NAMES),
        target_names=list(TARGET_NAMES), examples_by_split={s: len(arrays["X_"+s])
        for s in ("train", "val", "test", "test_action")})
    return arrays, metadata


def test_equal_banks_preserve_duration_pairs_and_cover_both_modes_across_shifts():
    rows = rows_for_context(); rows = [rows[i] for i in np.random.default_rng(4).permutation(18)]
    direct = audit.bank_indices(rows, "direct9")
    assert len(direct) == 9 and all(not rows[i]["reconfigure"] for i in direct)
    selected = []
    for shift in (0, 1):
        ids = audit.bank_indices(rows, f"allocated9_s{shift}")
        keys = [audit.action_key(rows[i]) for i in ids]
        assert len(set(keys)) == 9 and [key[1:] for key in keys] == list(audit.GRID)
        assert [key[0] for key in keys] == [bool((k+shift)%2) for k in range(9)]
        selected.append(set(keys))
    assert not selected[0] & selected[1]
    assert selected[0] | selected[1] == {audit.action_key(r) for r in rows}


def test_duplicate_or_incomplete_action_banks_are_rejected_before_selection():
    rows = rows_for_context()
    with pytest.raises(ValueError, match="Duplicate"):
        audit.bank_indices(rows+[deepcopy(rows[0])], "expanded18")
    with pytest.raises(ValueError, match="complete declared"):
        audit.bank_indices(rows[:-1], "direct9")


def test_context_and_metadata_validation_checks_common_state_and_array_order():
    data, meta = synthetic_dataset()
    blocks, checks = audit.validate_dataset(data, meta)
    assert len(blocks) == 2 and checks["decision_examples"] == 36
    assert checks["unseen_action_examples"] == 16
    changed = deepcopy(data); changed["X_test"][3, FEATURE_NAMES.index("motor_activation_0_0")] = .1
    with pytest.raises(ValueError, match="non-action state"):
        audit.validate_dataset(changed, meta)
    changed = deepcopy(data); changed["groups_test"][:18] = 201
    with pytest.raises(ValueError, match="context order"):
        audit.validate_dataset(changed, meta)


def test_metadata_label_tampering_and_holdout_program_duplicates_are_detected():
    data, meta = synthetic_dataset()
    changed = deepcopy(data); changed["Y_test"][0, 0] += .01
    with pytest.raises(ValueError, match="outcome labels"):
        audit.validate_dataset(changed, meta)
    changed_meta = deepcopy(meta); changed = deepcopy(data)
    rows = [r for r in changed_meta["rows"] if r["split"] == "test_action"]
    rows[1]["recovery_config"] = deepcopy(rows[0]["recovery_config"])
    changed["X_test_action"][1] = changed["X_test_action"][0]
    with pytest.raises(ValueError, match="eight declared"):
        audit.validate_dataset(changed, changed_meta)


def test_holdout_contexts_cannot_overlap_development_or_change_hidden_memory():
    data, meta = synthetic_dataset()
    changed_meta = deepcopy(meta); changed_meta["contexts"][2]["split"] = "train"
    with pytest.raises(ValueError, match="context order"):
        audit.validate_dataset(data, changed_meta)
    changed = deepcopy(data); changed["X_test_action"][0, FEATURE_NAMES.index("controller_payload_integral_x")] = .2
    with pytest.raises(ValueError, match="common state"):
        audit.validate_dataset(changed, meta)


def test_prediction_only_selection_is_fixed_before_actual_labels_are_revealed():
    predicted = np.array([[.02, .3, 7., .1, .8, .005, .01, 1.],
                          [.03, .4, 8., .1, .8, .005, .01, 1.]])
    selection = audit.select_predicted(predicted)
    assert selection["index"] == 0 and not selection["fallback"]
    truth = predicted.copy(); truth[0, 0] = .2
    scored = audit.score_selection(selection, predicted, truth, truth)
    assert selection["index"] == 0
    assert not scored["constraint_pass"] and scored["selected_false_predicted_feasible"]
    assert scored["bank_feasibility_miss"]
    assert scored["feasible_cost_regret_in_bank"] is None
    assert scored["best_true_feasible_cost_in_bank"] == pytest.approx(8.01)


def test_regret_distinguishes_bank_coverage_from_choice_quality_and_keeps_failures():
    truth = np.array([[.02, .3, 7., .1, .8, .005, .01, 1.],
                      [.03, .4, 8., .1, .8, .005, .01, 1.],
                      [.01, .2, 6., .1, .8, .005, .01, 1.]])
    predicted = truth[:2].copy(); predicted[:, 2] = [9., 7.]
    choice = audit.select_predicted(predicted)
    scored = audit.score_selection(choice, predicted, truth[:2], truth)
    assert scored["constraint_pass"] and scored["local_operation_completion"]
    assert scored["feasible_cost_regret_in_bank"] == pytest.approx(1.)
    assert scored["feasible_cost_regret_global"] == pytest.approx(2.)
    truth[:, 7] = 0.
    scored = audit.score_selection(choice, predicted, truth[:2], truth)
    assert not scored["local_operation_completion"] and not scored["constraint_pass"]
    assert scored["best_true_feasible_cost_global"] is None


def test_frozen_validation_margins_trigger_recorded_fallback_without_threshold_tuning():
    truth = np.array([[.02, .3, 7., .1, .8, .005, .01, 1.],
                      [.03, .4, 8., .1, .8, .005, .01, 1.]])
    choice = audit.select_predicted(truth, margins=(.06, .2))
    assert choice["fallback"] and not choice["feasible"].any()
    score = audit.score_selection(choice, truth, truth, truth)
    assert score["false_predicted_infeasible_count"] == 2
    assert score["constraint_pass"]
    bad = truth.copy(); bad[:, 2] = 18.01
    assert not audit.score_selection(choice, truth, bad, bad)["constraint_pass"]


def test_coverage_and_tied_rank_metrics_keep_joint_denominator_explicit():
    actual = np.zeros((3, 8)); predicted = actual.copy()
    actual[:, :2] = [[.01, .2], [.03, .4], [.08, .1]]
    coverage = audit.margin_coverage(predicted, actual, (.04, .3))
    assert coverage["excursion_covered"] == coverage["tension_covered"] == 2
    assert coverage["jointly_covered"] == 1 and coverage["examples"] == 3
    assert audit.rank_correlation([1., 1., 2.], [2., 2., 1.]) == pytest.approx(-1.)
    assert audit.rank_correlation([1., 1., 1.], [1., 2., 3.]) is None


def test_paired_counts_include_all_contexts_and_reject_missing_or_duplicate_decisions():
    rows = [dict(predictor="small/seed0", bank=bank, context_id=cid,
                 constraint_pass=bank != "direct9" and cid == 201)
            for bank in audit.BANKS for cid in (200, 201)]
    result = audit.paired_results(rows)
    assert all(r["contexts"] == 2 and r["improved"] == 1 and r["worsened"] == 0 for r in result)
    with pytest.raises(ValueError, match="share all"):
        audit.paired_results(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        audit.paired_results(rows+[deepcopy(rows[0])])


def test_frozen_reference_checks_fail_on_changed_input(tmp_path):
    path = tmp_path/"weights.npz"; path.write_bytes(b"frozen")
    references = {str(path): audit.sha(path)}
    audit.verify_references(references)
    path.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="hash changed"):
        audit.verify_references(references)
