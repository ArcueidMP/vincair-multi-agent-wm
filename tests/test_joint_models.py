import numpy as np
import pytest
import torch

from vmas_pusht.abstraction_models import ResponseMLP
from vmas_pusht.joint_audit import numpy_prediction
from vmas_pusht.joint_experiment import protocol, schedule
from vmas_pusht.joint_interface import INPUT_DIM
from vmas_pusht.joint_models import FrozenJointPredictor, train, validate
from vmas_pusht.joint_report import export_models, restore_models
from vmas_pusht.option_models import normalization


def dataset():
    g = torch.Generator().manual_seed(73)
    x = torch.randn(64, INPUT_DIM, generator=g)
    y = torch.rand(64, 5, generator=g)
    rows = [dict(context_id=f"context_{i}", split=("train", "val", "test_action", "test_context")[i//16],
                 maximum_norm=[.9, .9]) for i in range(64)]
    return dict(X=x, Y=y, rows=rows)


def test_context_partition_leakage_is_rejected():
    data = dataset()
    validate(data)
    data["rows"][16]["context_id"] = data["rows"][0]["context_id"]
    with pytest.raises(ValueError, match="Context crosses"):
        validate(data)


def test_numpy_forward_and_portable_weights_preserve_frozen_predictions(tmp_path):
    data = dataset()
    model = ResponseMLP(INPUT_DIM, 16, 5)
    entry = dict(width=16, normalization=normalization(data["X"][:16], data["Y"][:16]),
                 state_dict=model.state_dict(), duration_margin=.1, seed=0)
    pack = dict(format_version=1, checkpoints={0: entry})
    source = tmp_path/"source.pt"; torch.save(pack, source)
    exported = tmp_path/"weights.json"; restored = tmp_path/"restored.pt"
    export_models(source, exported); restore_models(exported, restored)
    recovered = torch.load(restored, weights_only=False)["checkpoints"][0]
    for field in ("normalization", "state_dict"):
        assert all(torch.equal(entry[field][k], recovered[field][k]) for k in entry[field])
    a = FrozenJointPredictor(entry)(data["X"])
    b = FrozenJointPredictor(recovered)(data["X"])
    assert torch.equal(a, b)
    np.testing.assert_allclose(a.numpy(), numpy_prediction(entry, data["X"].numpy()), atol=1e-5, rtol=0.)
    with pytest.raises(FileExistsError):
        restore_models(exported, restored)


def test_training_uses_only_train_moments_and_a_fixed_checkpoint(tmp_path):
    torch.set_num_threads(1)
    data = dataset()
    data["X"][16:] += 200.
    source = tmp_path/"data.pt"; torch.save(data, source)
    result = train([source], tmp_path/"models.pt", steps=3, seeds=(0,), width=8)
    pack = torch.load(tmp_path/"models.pt", weights_only=False)
    expected = normalization(data["X"][:16], data["Y"][:16])
    actual = pack["checkpoints"][0]["normalization"]
    assert all(torch.equal(actual[k], expected[k]) for k in actual)
    assert result["training"]["steps"] == 3
    assert result["training"]["final_checkpoint"]
    assert result["splits"] == dict(train=16, val=16, test_action=16, test_context=16)


def test_prespecified_matrix_counts_independent_physics_and_model_repeats_correctly():
    jobs = schedule()
    assert len(jobs) == len(set(jobs)) == 64
    assert sum(j[0] == "oracle" for j in jobs) == 16
    assert all(j[1] == 0 for j in jobs if j[0] == "oracle")
    assert len(jobs)*18 == protocol()["cases"] == 1152
    assert protocol()["joint_candidate_budget"] == 7*protocol()["fixed_candidate_budget"]
