import pytest

from vmas_pusht.interface_experiment import jobs
from vmas_pusht.option_planner import MODEL_NAMES, PLANNERS


def test_matrix_keeps_all_model_seeds_but_does_not_repeat_physical_controls():
    schedule=jobs()
    assert len(schedule)==36
    assert len(schedule)==len(set(schedule))
    for planner in PLANNERS:
        selected=[(seed,goal) for name,seed,goal in schedule if name==planner]
        expected_seeds={0,1,2} if planner in MODEL_NAMES else {0}
        assert {seed for seed,_ in selected}==expected_seeds
        assert {goal for _,goal in selected}=={"standard","holdout"}
        assert len(selected)==2*len(expected_seeds)


def test_duplicate_matrix_factors_cannot_inflate_case_counts():
    with pytest.raises(ValueError,match="Duplicate"):
        jobs(("wm_full","wm_full"))
    with pytest.raises(ValueError,match="Duplicate"):
        jobs(model_seeds=(0,0))
