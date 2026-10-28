# W-ar backbone interface to agree with Yihao

W-ar owns the physical-state head, multistep state loss, training loop and arena
adapter. Yihao owns the shared entity tokenizer and transformer, including the
`full` attention mask. The proposed import contract is:

```python
backbone = factory(**backbone_options)  # torch.nn.Module
backbone.token_dim: int
tokens = backbone(state_history, recent_setpoints, recent_times, setpoint, age)
# tokens.shape == (batch, 4, backbone.token_dim)
```

Inputs are `float32` tensors on the backbone's device:

| Input | Shape | Meaning |
| --- | --- | --- |
| `state_history` | `(B, history, 51)` | normalized physical states, oldest first; 3 aircraft × 15 values, then payload position/velocity |
| `recent_setpoints` | `(B, 4, 3, 3)` | last four setpoints, relative to the newest observed payload position |
| `recent_times` | `(B, 4)` | those setpoints' start times relative to the current decision, in seconds |
| `setpoint` | `(B, 3, 3)` | current candidate segment endpoint, relative to the newest observed payload position |
| `age` | `(B,)` | age of the delayed observation in seconds; zero after the first predicted step |

Token order is aircraft 0, 1, 2, payload. The head maps those four tokens to
normalized physical-state increments. No latent target or SIGReg is used. The
first increment predicts across the observation's age plus one decision. The
same module rolls out all 15 decisions, keeping gradients through every step.
The backbone should attach each aircraft's action inputs to that aircraft's
token and use the JEPA `full` attention mask, so the backbone is identical to
the `full` variant apart from W-ar's direct-state objective and head.

The factory is passed by `--backbone module:factory --backbone-options '{...}'`
when training. The training CLI defaults to the standard
`data/uav_arena/datasets/explore_v2` dataset. Its import path and options are
stored in the checkpoint so
`uav_arena.participants.war.model:make` can rebuild it for evaluation. The
options must select the JEPA `full` attention mask and match its history and
parameter count as closely as practical. The exact factory path and options
need agreement with Yihao before W-ar's Phase A training.
