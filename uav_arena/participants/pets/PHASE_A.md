# June's Phase A progress report

Status: the PETS Phase A bar is met on `explore-v2`. Work is on
`uav/june-pets-w-ar`, rebased onto `origin/uav-arena` at the frozen
`arena-v0.3` contract. W-ar's outer loop remains ready for Yihao's shared
backbone.

## Data

The downloaded standard datasets passed the frozen content check:

| Dataset | Match | Differ | Unknown |
| --- | ---: | ---: | ---: |
| `explore_v2` | 2,000 | 0 | 0 |
| `explore_v2_val` | 100 | 0 | 0 |

The arena and shared-backbone regression suite produced 95 passed, 2 skipped
and 1 deselected. The deselected test regenerates an episode and compares its
exact content hash;
the task card documents that Ubuntu 22.04 does not reproduce the frozen
Ubuntu 26.04 reference. No frozen file was changed to hide that platform
difference, and the copied standard datasets themselves match every manifest.

All Phase A training below uses only the first 250 training episodes. Model
selection uses all 100 validation episodes with `openloop.py --stride 10` and
does not use `dev` outcomes.

## PETS selection

Each member is a four-hidden-layer Swish MLP that predicts a bounded diagonal
Gaussian over the next 51-dimensional physical-state increment. Five members
train on independent bootstrap resamples with Gaussian NLL. At inference, a
member is drawn per candidate and held for the whole rollout (one-particle
TS-infinity), but the member's mean is propagated without Gaussian sampling.

| Configuration | Width | Steps/member | Gaussian sampling | Selection score |
| --- | ---: | ---: | :---: | ---: |
| PR #2 configuration, maintainer rerun | 200 | 5,000 | yes | 18.60 cm |
| PR #2 configuration, maintainer rerun | 200 | 5,000 | no | 9.42 cm |
| Phase A selected | 256 | 10,000 | no | **7.12 cm** |
| Example Phase A bar | 512 | 5,000 | no | **8.26 cm** |

The maintainer retrained this selected configuration on Ubuntu 26.04 with an
RTX 5070 Laptop and reproduced the selection score, every per-latency payload
and aircraft error, and the parameter count to the reported precision. The
Phase A review passed; the PR remains a draft through Phase B.

Before `arena-v0.3` replaced the standard data, the following development
configurations were also scored on the older `explore-v1_val`. They are listed
for completeness but are not comparable with, and did not select, the v2 row:

| Legacy `explore-v1` configuration | Steps/member | Selection score |
| --- | ---: | ---: |
| Single-frame prototype | 1,000 | 21.51 cm |
| Four-frame history, fixed observed-payload origin | 2,000 | 15.19 cm |
| Four-frame history, current-payload centering | 2,000 | 5.66 cm |
| Four-frame history, current-payload centering | 5,000 | 5.23 cm |

The selected configuration's validation errors are:

| Latency | Payload | Aircraft |
| --- | ---: | ---: |
| 0 ms | 5.89 cm | 5.40 cm |
| 50 ms | 6.43 cm | 5.90 cm |
| 100 ms | 7.20 cm | 6.71 cm |
| 200 ms | 8.95 cm | 8.34 cm |

It has 1,444,860 parameters and trained in 665.75 s on an RTX 5060 Laptop GPU
(about 0.185 GPU-hours). Across the selected model's `dev` flight, a
512-candidate, 15-decision planning call took 23.4 ms on average.

The shared planner accepts one trajectory per candidate. This row therefore
deviates from published PETS in two stated ways: it propagates one particle
instead of 20, and it uses the sampled member's Gaussian mean instead of
sampling a state increment. Member selection remains random, so the predictor
declares `stochastic = True`, draws from torch's global generator and reseeds
that generator in `reset()`.

## One `dev` flight after selection

The selected model was flown on `dev` once, after it passed the open-loop bar.
It succeeded on **7/40** episodes.

| Condition | Success |
| --- | ---: |
| static / 0 ms | 1/5 |
| static / 50 ms | 1/5 |
| static / 100 ms | 1/5 |
| static / 200 ms | 0/5 |
| static+moving / 0 ms | 2/5 |
| static+moving / 50 ms | 0/5 |
| static+moving / 100 ms | 1/5 |
| static+moving / 200 ms | 1/5 |

The 33 failures were 18 team-separation events, 12 obstacle collisions, two
timeouts and one crash. `dev` was not used to select another configuration.

## W-ar and remaining work

`participants/war/` contains a direct physical-state increment head, a
15-decision differentiable state loss, the training CLI and the planner
adapter. Its training data path defaults to `explore_v2`. Yihao's shared
tokenizer/transformer is now integrated at
`uav_arena.participants.jepa.backbone:make_backbone`, and tests exercise both
the stand-in and real backbone. The interface and field-major state layout are
confirmed in `participants/war/README.md`. Formal W-ar training waits for the
JEPA full row's final structural configuration and total capacity so the two
remain matched.

Checkpoints and raw JSON outputs remain in the ignored `data/uav_arena/` tree
as required by `PARTICIPANTS.md`; only the code, tests, commands and this
summary are committed. Phase B has not begun.
