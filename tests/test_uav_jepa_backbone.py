"""Shared-backbone contract, information flow and differentiable rollout checks."""
import importlib

import pytest

torch = pytest.importorskip("torch")

from uav_arena.participants.jepa.backbone import UAVBackbone, pack_state, split_state


def small(attn="full", history=4):
    torch.manual_seed(7)
    return UAVBackbone(dim=16, latent=8, enc_layers=1, pred_layers=1, heads=4,
                       history=history, attn=attn)


def inputs(batch=2, history=4):
    generator = torch.Generator().manual_seed(12)
    return (torch.randn(batch, history, 51, generator=generator),
            torch.randn(batch, 4, 3, 3, generator=generator),
            torch.tensor([-.3, -.2, -.1, 0.]).expand(batch, -1).clone(),
            torch.randn(batch, 3, 3, generator=generator), torch.full((batch,), .2))


def test_field_major_state_is_split_into_the_correct_aircraft_features():
    state = torch.arange(51, dtype=torch.float32)[None, None].expand(2, 4, -1)
    aircraft, payload = split_state(state)
    for i in range(3):
        expected = torch.tensor([*range(3*i, 3*i+3), *range(9+3*i, 12+3*i),
                                 *range(18+6*i, 24+6*i), *range(36+3*i, 39+3*i)])
        torch.testing.assert_close(aircraft[0, 0, i], expected.float())
    torch.testing.assert_close(payload[0, 0], state[0, 0, 45:])


def test_packing_uses_one_payload_origin_and_never_reads_metadata():
    class State(dict):
        def __getitem__(self, key):
            assert key not in ("payload_mass", "scenario", "layout", "seed")
            return super().__getitem__(key)

    state = State(uav_pos=torch.randn(2, 4, 3, 3), uav_vel=torch.randn(2, 4, 3, 3),
                  uav_rot6=torch.randn(2, 4, 3, 6), uav_omega=torch.randn(2, 4, 3, 3),
                  payload_pos=torch.randn(2, 4, 3), payload_vel=torch.randn(2, 4, 3),
                  payload_mass=object(), scenario=object())
    origin = state["payload_pos"][:, -1:]
    packed = pack_state(state, origin)
    assert packed.shape == (2, 4, 51)
    torch.testing.assert_close(packed[:, -1, 45:48], torch.zeros(2, 3))
    offset = torch.tensor([3., -2., 1.])
    shifted = State(state)
    shifted["uav_pos"] = state["uav_pos"]+offset
    shifted["payload_pos"] = state["payload_pos"]+offset
    torch.testing.assert_close(pack_state(shifted, origin+offset), packed, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
@pytest.mark.parametrize("batch", [1, 512, 555])
def test_factory_and_shapes_for_planner_and_harness_batches(attn, batch):
    factory = importlib.import_module("uav_arena.participants.jepa.backbone").make_backbone
    model = factory(dim=16, latent=8, enc_layers=1, pred_layers=1, heads=4, history=4, attn=attn).eval()
    values = inputs(batch)
    with torch.no_grad():
        output = model(*values)
        again = model(*values)
    assert output.shape == (batch, model.n_tok, model.token_dim)
    assert output.dtype == values[0].dtype and output.device == values[0].device
    assert torch.isfinite(output).all()
    torch.testing.assert_close(again, output, atol=0, rtol=0)


def test_flat_matches_latent_budget_and_full_no_cross_match_parameters():
    full, no_cross, flat = (small(a) for a in ("full", "no_cross", "flat"))
    assert full.latent_budget == no_cross.latent_budget == flat.latent_budget == 32
    assert full.token_dim == 8 and flat.token_dim == 32
    assert sum(p.numel() for p in full.parameters()) == sum(p.numel() for p in no_cross.parameters())


def test_masks_are_causal_and_no_cross_preserves_the_payload_route():
    model = small("no_cross")
    for query in range(16):
        for key in range(16):
            qi, ki = query % 4, key % 4
            blocked = key//4 > query//4 or (qi < 3 and ki < 3 and qi != ki)
            assert bool(model.pred_mask[query, key]) == blocked
    assert not model.enc_mask[0, 3] and not model.enc_mask[3, 0]


def test_each_action_is_attached_only_to_its_aircraft_before_attention():
    model = small()
    _, recent, times, setpoint, age = inputs()
    original = model._action_tokens(recent, times, setpoint, age)
    changed = setpoint.clone()
    changed[:, 1] += 2.
    difference = model._action_tokens(recent, times, changed, age)-original
    torch.testing.assert_close(difference[:, [0, 2, 3]], torch.zeros_like(difference[:, [0, 2, 3]]))
    assert difference[:, 1].abs().max() > 0


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
def test_future_latents_and_current_action_do_not_change_older_outputs(attn):
    model = small(attn).eval()
    state, recent, times, setpoint, age = inputs()
    z = model.encode(state)
    original = model._predict_history(z, recent, times, setpoint, age)
    changed = z.clone()
    changed[:, -1] += torch.randn_like(changed[:, -1])
    future_changed = model._predict_history(changed, recent, times, setpoint, age)
    action_changed = model._predict_history(z, recent, times, setpoint+2., age)
    torch.testing.assert_close(future_changed[:, :-1], original[:, :-1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(action_changed[:, :-1], original[:, :-1], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(action_changed[:, -1], original[:, -1])


def test_one_no_cross_layer_blocks_direct_action_influence_but_not_payload():
    model = small("no_cross").eval()
    state, recent, times, setpoint, age = inputs()
    original = model(state, recent, times, setpoint, age)
    changed = setpoint.clone()
    changed[:, 1] += 2.
    output = model(state, recent, times, changed, age)
    torch.testing.assert_close(output[:, 0], original[:, 0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(output[:, 3], original[:, 3])


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
def test_gradients_reach_state_actions_times_age_and_all_parameters(attn):
    model = small(attn)
    values = [value.requires_grad_() for value in inputs()]
    loss = model(*values).square().mean()
    loss.backward()
    for value in values:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().max() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_fifteen_step_latent_rollout_keeps_gradient_to_the_first_action():
    model = small()
    state, recent, times, setpoint, age = inputs()
    future = torch.randn(2, 15, 3, 3, requires_grad=True)
    history = model.encode(state)
    for k in range(15):
        next_z = model.predict(history, recent, times, future[:, k], age)
        history = torch.cat((history[:, 1:], next_z[:, None]), dim=1)
        recent = torch.cat((recent[:, 1:], future[:, k:k+1]), dim=1)
        times = torch.cat((times[:, 1:]-.1, torch.full_like(times[:, :1], -.1)), dim=1)
        age = torch.zeros_like(age)
    next_z.square().mean().backward()
    assert torch.isfinite(future.grad).all() and future.grad[:, 0].abs().max() > 0


@pytest.mark.parametrize("attn", ["full", "no_cross", "flat"])
def test_checkpoint_rebuild_is_identical_and_short_history_is_supported(attn, tmp_path):
    model = small(attn).eval()
    checkpoint = tmp_path/"backbone.pt"
    torch.save(dict(config=model.config, state=model.state_dict()), checkpoint)
    pack = torch.load(checkpoint, weights_only=True)
    rebuilt = UAVBackbone(**pack["config"]).eval()
    rebuilt.load_state_dict(pack["state"])
    values = inputs(history=1)
    torch.testing.assert_close(rebuilt(*values), model(*values), atol=0, rtol=0)


@pytest.mark.parametrize("attn", ["unknown", "", "FULL"])
def test_unknown_attention_variant_is_rejected(attn):
    with pytest.raises(ValueError, match="attn"):
        small(attn)


def test_wrong_state_layout_or_context_shape_is_rejected():
    model = small()
    state, recent, times, setpoint, age = inputs()
    with pytest.raises(ValueError, match="51"):
        model(state[..., :50], recent, times, setpoint, age)
    with pytest.raises(ValueError, match="history length"):
        model(inputs(history=5)[0], recent, times, setpoint, age)
    with pytest.raises(ValueError, match="action context shape"):
        model(state, recent[:, :3], times, setpoint, age)


def test_float64_inputs_and_model_preserve_dtype():
    model = small().double()
    output = model(*(value.double() for value in inputs()))
    assert output.dtype == torch.float64 and torch.isfinite(output).all()
