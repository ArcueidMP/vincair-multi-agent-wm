# Shared UAV backbone and JEPA participant

Factory: **`uav_arena.participants.jepa.backbone:make_backbone`**.
This module adapts `vmas_pusht/model.py`'s type-specific token embeddings,
frame encoder, latent bottleneck and causal action-conditioned predictor to
the arena. `backbone.py` contains no JEPA loss, SIGReg, state head or training
loop; W-ar supplies its own direct-state head and loss. The JEPA-specific
objective, adapter and training entrypoint are separate modules described below.

## Task and evaluation

This participant implements Yihao's three JEPA rows and the shared W-ar backbone
assigned in [TASKS.md](../../TASKS.md). The arena's [README](../../README.md)
defines the shared MPPI; [ANALYSIS.md](../../ANALYSIS.md) defines the comparison
of cross-agent prediction error with closed-loop success.

| File | Responsibility |
|---|---|
| `backbone.py` | UAV/payload tokenizer, encoder, action-conditioned predictor and attention variants; imported by W-ar |
| `objective.py` | Latent rollout, SIGReg and the 15-feature physical readout loss |
| `model.py` | Arena factory, observation history, reset and batched trajectory prediction |
| `train.py` | Training-only normalization, optimization, checkpoints and run metadata |
| [PHASE_A.md](PHASE_A.md) | Seven completed runs, reproduction commands, compute and remaining tasks |
| [DIAGNOSTICS_FULL_D.md](DIAGNOSTICS_FULL_D.md) | C/D/example error comparison, frozen readout probes and normalization discussion |
| `data/uav_arena/results/jepa/phase_a_summary.json` | Local configurations, metrics and hashes |

The factory `uav_arena.participants.jepa.model:make` serves both the shared
planner and the [evaluation harness](../../HARNESS.md). It accepts single-plan
`(1, 1)`, MPPI `(1, 512)` and harness `(1, 555)` leading shapes, followed by
`(15, 3, 3)` setpoints. It returns aircraft positions, payload positions and
payload velocities at each decision endpoint.

Phase A uses 250 training episodes and full validation at stride 10. The full
variant must score strictly below 0.082593 m (8.2593 cm); this qualifies the shared
codebase, not each variant separately. The full seed-0 run of 40,000 steps completed with 8.4265 cm, above the gate;
the best full result remains D (23,000 steps, 8.3737 cm). Both used random
initialization and the same recipe. Report E and await maintainer guidance;
do not add training or change seeds. Target-scale changes remain deferred.
After full passes, run dev once per variant (40 episodes): full uses the 40,000-step
checkpoint; no_cross and flat reuse their existing C checkpoints without retraining.
Dev checks closed-loop behavior and does not select configurations. The maintainer
reproduces full and decides Phase B entry. See [PHASE_A.md](PHASE_A.md).

## W-ar import contract

```python
from uav_arena.participants.jepa.backbone import make_backbone

backbone = make_backbone(history=4, attn="full")
tokens = backbone(state_history, recent_setpoints, recent_times, setpoint, age)
# tokens: (B, 4, backbone.token_dim), aircraft 0, 1, 2, payload
```

Pass the same `history` in W-ar's CLI and `--backbone-options`. Defaults:
`history=4, dim=128, latent=64, enc_layers=2, pred_layers=4, heads=4`.
Dropout is zero. `backbone.config` is sufficient to rebuild the architecture;
save it with `state_dict`. Normalization statistics belong to the caller's
checkpoint, not the backbone. Do not pass mass, layout or simulator state.

| Argument | Shape | Units and meaning |
|---|---|---|
| `state_history` | `(B, H, 51)`, `1 <= H <= history` | Normalized physical states, oldest first; field-major order below |
| `recent_setpoints` | `(B, 4, 3, 3)` | Metres relative to the latest observed payload; padded with the hover endpoint at episode start |
| `recent_times` | `(B, 4)` | Endpoint command start times relative to the current decision, in seconds |
| `setpoint` | `(B, 3, 3)` | Next candidate segment endpoint, metres relative to the same origin |
| `age` | `(B,)` | Latest observation's actual age in seconds, not always the nominal latency |

Inputs must share the module's device and floating dtype (W-ar uses float32).
The backbone preserves them. It is stateless: episode history and `reset()`
are owned by the participant adapter. No future state is an input.

**State layout:** the backbone uses the same **field-major** 51-vector as
W-ar's `participants/state.py::pack_state`:

| Slice | Contents |
|---|---|
| `0:9` | All three aircraft positions |
| `9:18` | All three aircraft velocities |
| `18:36` | All three aircraft rotation columns (6 per aircraft) |
| `36:45` | All three aircraft body rates |
| `45:48` | Payload position |
| `48:51` | Payload velocity |

This factory follows the executable layout. `split_state` assembles each
aircraft's 15 features from those slices. `pack_state(state, origin)` is a
compatible utility, reading only six named state fields. For a history,
use the same newest observed payload origin for **all** frames; do not centre
each historical frame separately. Fit moments only on training data.

## Delays, actions and rollout

The first requested transition ends one decision after the current decision
time: its prediction spans `age + 0.1 s` from the latest observation. The
backbone receives age and all four past commands; the objective must use the
matching target supplied by `windows.py`, without an extra index shift.
This is a learned delay bridge, not a simulator or a hard-coded integrator.

Each aircraft's four past endpoints, their times, its candidate endpoint and
age form that aircraft's 20-value action input. The payload has no direct
action embedding. Action context enters the newest history frame only;
the transformer uses a causal time mask, so older outputs cannot see it.

- **W-ar:** call `forward` after every physical-state update; it re-encodes
  normalized predicted states. Its head predicts state increments. W-ar's
  existing rollout updates recent commands/times and sets age to zero after
  the first transition. Keep gradients through the whole rollout.
- **JEPA:** use `encode(state)` for observed/target latents, and
  `predict(latent_history, recent_setpoints, recent_times, setpoint, age)`
  for latent autoregression. Append the predicted latents to the history,
  update the action context on each step, then decode with JEPA's readout.
  SIGReg belongs to JEPA training only; prediction itself is deterministic.

The two objectives share the actual tokenizer/transformer implementation,
not trained weights. W-ar uses `attn="full"`. For paired experiments,
align history and width and report total parameter counts. The JEPA readout
predicts 15 planner features; W-ar predicts all 51 state features. JEPA parameter
counts are in [PHASE_A.md](PHASE_A.md).

## Variants and capacity

- `full`: four tokens, all entity pairs visible at permitted times.
- `no_cross`: four tokens, aircraft queries cannot directly attend to another
  aircraft; self and payload remain visible. Multiple layers can transmit
  cross-aircraft information **through the payload**. This is not complete
  causal independence.
- `flat`: one token per frame. `latent` means per-entity width in the
  structured variants, so the flat token automatically has width `4*latent`.
  All variants have `latent_budget = n_tok * token_dim = 4*latent`.
  Hidden transformer `dim` stays fixed; parameter counts can differ and must
  be reported. Flat is not compatible with W-ar's four-token state head.

`full` and `no_cross` have identical learned parameter shapes; only their
attention masks differ. The encoder and predictor masks both enforce the
variant. The predictor retains the upstream learned token-slot and time embeddings.

## Checks and handoff

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  data/uav_arena/.venv/bin/python -m pytest -q tests/test_uav_jepa_backbone.py
```

The tests cover field order, payload-relative packing, excluded metadata,
candidate batch sizes 1/512/555, action ownership, causal masks and numeric
causal isolation, gradient flow over 15 steps, deterministic repeated calls,
checkpoint reconstruction and invalid shapes.

Suggested W-ar arguments, once this module is available in june's checkout:

```bash
--backbone uav_arena.participants.jepa.backbone:make_backbone \
--history 4 \
--backbone-options '{"history":4,"attn":"full","dim":128,"latent":64,"enc_layers":2,"pred_layers":4,"heads":4}'
```

Integration was checked against W-ar revision
`1e96f2ec37c749053b0cd0e4e50f29d6df3e4093` on `uav/june-pets-w-ar`,
including 15-step backpropagation and checkpoint predictions for 1/512/555
candidates. The default backbone history is 4; the current Phase A C runs
use history 1. Pass the same history in both W-ar arguments when comparing.

## JEPA training and prediction

`objective.py` adds the upstream-style latent MSE, SIGReg, and a readout loss
on both encoded and autoregressively predicted latents. The readout predicts
the planner's 15 required values (nine aircraft positions, payload position
and velocity), not all 51 input features. Its absolute and delta losses are
normalized with training-only state and increment statistics. Gradients flow
through the encoder, predictor and readout; targets are not EMA/detached.

The first transition spans the actual observation age plus one decision;
later transitions span 0.1 s. A rollout uses only observation latents and
actions, never future encoded targets. SIGReg samples projections in training
only; the published participant is deterministic and does not declare itself
stochastic.

Phase A configuration A (measured outcomes are in [PHASE_A.md](PHASE_A.md)):

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 250 --steps 5000 \
  --batch 128 --head-weight 5 --lr 0.001 --attn full \
  --out data/uav_arena/participants/jepa/my_full_a.pt
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/my_full_a.pt"}' \
  --data data/uav_arena/datasets/explore_v2_val --stride 10 \
  --out data/uav_arena/results/my_full_a_openloop.json
```

Every run needs a fresh output prefix. The trainer writes training arguments,
selected training seeds, the dataset manifest hash, source hashes, parameter
count, loss components and runtime alongside the checkpoint. Intermediate
`.stepN.pt` files are labelled incomplete for debugging. Configuration
selection uses the validation score; dev runs check closed-loop behavior.

`model.py::make(checkpoint, device=None)` is the participant entrypoint.
Default device is CUDA when available, otherwise CPU; returned predictions
always match the candidate tensor's leading shape, device and dtype. It
caches only observation encodings. `reset()` clears every observed frame and
cache, and candidate rollouts never change the observed history. `layout`
and `info` metadata are not read.

Run participant and backbone tests together:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  data/uav_arena/.venv/bin/python -m pytest -q \
  tests/test_uav_jepa_backbone.py tests/test_uav_jepa_participant.py
```

After Phase A review, each variant selects its configuration on the full
validation set and trains three new seeds at each of the 2,000/500-episode
budgets. [PHASE_A.md](PHASE_A.md) tracks the results and next tasks.


## Frozen encoder readout probe

Fit a fresh readout while keeping the complete parent model frozen:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.probe_readout \
  --checkpoint data/uav_arena/participants/jepa/phase_a_full_c.pt \
  --device cuda --out data/uav_arena/results/jepa/full_c_readout_probe_v1
```

Defaults: first 250 training episodes, 5,000 steps, batch 128, learning rate
0.001, seed 0, AdamW and OneCycle. The probe reuses the parent's normalization
and optimizes only the encoded-state readout MSE. It caches all training-window
true-frame encodings on CPU, then evaluates the original/new heads on every
10th decision of all 100 validation episodes at all four latencies.

The diagnostic reports observed and future aircraft/payload position errors
in cm and payload velocity errors in m/s, including steps 1–5/6–10/11–15.
The parent model and checkpoint hashes are checked after training. Probe
weights use a separate `.probe.pt` file and are not loaded by the participant
factory. See [PHASE_A.md](PHASE_A.md) for the measured reconstruction results.


### 关机后的续训与完整断点

`continue_train.py` 将旧的仅含权重的检查点作为新实验的起点，保留模型、损失和训练统计量，重置 AdamW / OneCycleLR，并重新开始确定性的训练批次顺序。它不等价于恢复旧实验的优化器轨迹。训练仍限于 Phase A 的前 250 个标准训练 episodes。

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.continue_train \
  --checkpoint data/uav_arena/participants/jepa/phase_a_full_d.step10000.pt \
  --training-config data/uav_arena/participants/jepa/phase_a_full_c.json \
  --steps 13000 --device cuda --workers 4 \
  --out data/uav_arena/participants/jepa/phase_a_full_d_warmstart_v1.pt
```

每 100 步原子替换同名前缀的 `.resume.pt`，保存权重、优化器、调度器、CPU/CUDA 随机状态、已完成批次位置和累计训练用时。`SIGINT` / `SIGTERM` 会在当前更新完成后保存退出；突然断电最多丢失最近 99 步。数据窗口读取是确定性的，worker 预取不会推进保存的批次位置。

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.continue_train \
  --resume data/uav_arena/participants/jepa/phase_a_full_d_warmstart_v1.resume.pt \
  --device cuda --workers 4 \
  --out data/uav_arena/participants/jepa/phase_a_full_d_warmstart_v1.pt
```

恢复时沿用检查点中的配置和总步数，核对训练源码与数据清单哈希。连续与分段运行的精确一致性已有 CPU 回归测试；CUDA 算子的数值复现仍受运行环境影响。最终 `.pt` 可由现有参赛者接口加载，`.resume.pt` 仅供训练恢复。


### 从头训练的完整断点

`train.py` 从随机初始化训练，并保持原 `RandomSampler` 的批次顺序。每 100 步原子保存 `.resume.pt`，包含模型、优化器、学习率计划、CPU/CUDA 随机状态、loader 生成器、当前 epoch 批次列表和已完成位置。`SIGINT` / `SIGTERM` 在当前更新结束后保存退出。恢复命令：

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.train \
  --resume data/uav_arena/participants/jepa/phase_a_full_d_fresh_v1.resume.pt \
  --out data/uav_arena/participants/jepa/phase_a_full_d_fresh_v1.pt \
  --device cuda --workers 4
```

恢复沿用保存的训练配置，必须保持 device 和 workers 一致；源码或数据清单发生变化会拒绝恢复。CPU 回归测试覆盖原始 shuffle 顺序以及跨 epoch、worker 预取条件下的恢复一致性。`full_d_fresh_v1` 从第 0 步开始，目标 23,000 步，与 `full_d_warmstart_v1` 分开保存。
