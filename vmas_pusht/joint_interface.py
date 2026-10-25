"""Joint primary/helper options with bounded, physically executed feedback.

A joint option is [route, direction, strength, push_steps, helper_id]. Both
agents' options are chosen together and run concurrently, including navigation.
The helper bank changes actuator commands/reference memory, never body state.
Zero helper id reproduces the historical fixed-support controller exactly.
"""
from dataclasses import fields

import torch

from vmas_pusht.action_interface import ActionInterface, candidate_table, model_inputs
from vmas_pusht.boundary import _rotate
from vmas_pusht.physical_abstraction import helper_action, norm_bound
from vmas_pusht.stable_local_wm import full_body_features


HELPERS = ("fixed_support", "release", "track_contact", "press_corner",
           "withdraw_corner", "slide_positive", "slide_negative")
# Offsets from the remembered anchor, in the bar frame at option start.
OFFSETS = ((0., 0.), (0., 0.), (0., 0.), (-.025, .025),
           (.04, -.04), (.04, .04), (-.04, -.04))
INPUT_DIM = 38 + len(HELPERS) + 1


def joint_table(*, planned_helper=True, expanded=False, allocation=0):
    """39 primary programs x all seven helper options (or fixed helper only).

Expanded tables preserve this exact primary-program/helper multiset; each
pair receives one of three routes by a fixed Latin allocation. All zero pushes
stay direct. Three allocations cover every route for every nonzero pair.
Thus direct/expanded have 273 distinct slots when jointly planning, 39 when
helper is fixed. Fixed-versus-planned compute is NOT claimed equal.
"""
    if allocation not in (0, 1, 2):
        raise ValueError("Expected prespecified allocation 0, 1 or 2")
    primary = candidate_table(expanded=False)
    helpers = torch.arange(len(HELPERS) if planned_helper else 1)
    table = torch.cat((primary.repeat_interleave(len(helpers), 0),
                       helpers.repeat(len(primary))[:, None]), 1)
    if expanded:
        for i in range(len(primary)):
            duration, pulse = divmod(i, 13)
            if pulse:
                sl = slice(i*len(helpers), (i+1)*len(helpers))
                table[sl, 0] = (duration+pulse-1+helpers+allocation) % 3
    return table


def validate_options(value):
    if value.ndim not in (2, 3) or value.shape[-1] != 5:
        raise ValueError("Expected B,5 or B,K,5 joint options")
    h = value[..., 4]
    if not torch.isfinite(value).all() or not ((h == h.floor()) & (h >= 0) & (h < len(HELPERS))).all():
        raise ValueError("Expected finite options with declared helper IDs")


def features(env, interface, options):
    """Goal-free physical inputs, current anchor/mode, and both agent choices."""
    validate_options(options)
    coarse, full = model_inputs(env, interface.anchor, interface.mode,
                               options[..., :4], reference_substeps=160)
    shape = options.shape[:-1]
    base = torch.cat((full_body_features(coarse[..., :18].reshape(-1, 18),
                                       full[..., :32].reshape(-1, 32)),
                      coarse[..., 18:].reshape(-1, 6)), 1).reshape(*shape, 38)
    helper = torch.nn.functional.one_hot(options[..., 4].long(), len(HELPERS)).to(base)
    active = base.new_tensor([s != "passive" for s in interface.supports])
    active = active.reshape(len(active), *([1]*(len(shape)-1)), 1).expand(*shape, 1)
    return torch.cat((base, helper, active), -1)


class JointActionInterface(ActionInterface):
    _tensor_fields = ActionInterface._tensor_fields + ("helper_id", "helper_target")

    def __init__(self, env, supports, anchor, *, config=None):
        super().__init__(env, supports, anchor, config=config)
        self.helper_id = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.helper_target = self.anchor.clone()

    def valid_options(self, options):
        validate_options(options)
        return super().valid_options(options[..., :4])

    def start(self, options, mask=None):
        value = torch.as_tensor(options, dtype=self.anchor.dtype, device=self.env.device)
        validate_options(value)
        if value.shape != (self.env.num_envs, 5):
            raise ValueError("Expected one joint option per environment")
        mask = self._mask(mask)
        # Parent validates physical options before any helper memory changes.
        result = super().start(value[:, :4], mask)
        self.helper_id[mask] = value[mask, 4].long()
        offsets = self.anchor.new_tensor(OFFSETS)[self.helper_id]
        target = self.anchor + _rotate(offsets, self.env.scenario.bar.state.rot[:, 0])
        self.helper_target[mask] = target[mask]
        return result

    def command(self, active_mask=None):
        command = super().command(active_mask)
        active = self.busy & self._mask(active_mask)
        helper = self.env.world.agents[1]
        fixed = command[:, 1].clone()
        reference = norm_bound(30.*(self.helper_target-helper.state.pos)-4.*helper.state.vel)
        track = helper_action(self.env, "track_contact", self.anchor)
        selected = torch.where((self.helper_id == 0)[:, None], fixed, reference)
        selected[self.helper_id == 1] = 0.
        selected[self.helper_id == 2] = track[self.helper_id == 2]
        command[:, 1] = selected*active[:, None]
        return command

    def reset_contact(self, mask, *, anchor=None, mode=3):
        result = super().reset_contact(mask, anchor=anchor, mode=mode)
        mask = self._mask(mask)
        self.helper_id[mask] = 0
        self.helper_target[mask] = self.anchor[mask]
        return result

    def clone_to(self, env, indices=None):
        idx = torch.arange(self.env.num_envs) if indices is None else torch.as_tensor(indices, dtype=torch.long)
        if idx.ndim != 1 or len(idx) != env.num_envs or ((idx < 0) | (idx >= self.env.num_envs)).any():
            raise ValueError("One valid source memory index per environment is required")
        result = JointActionInterface(env, [self.supports[i] for i in idx.tolist()], self.anchor[idx].to(env.device),
                                      config=self.state.config)
        result.state = self.state.clone(idx)
        for field in fields(result.state):
            value = getattr(result.state, field.name)
            if torch.is_tensor(value):
                setattr(result.state, field.name, value.to(env.device).clone())
        for name in self._tensor_fields:
            setattr(result, name, getattr(self, name)[idx].to(env.device).clone())
        return result
