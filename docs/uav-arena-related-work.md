# UAV arena: related work

Related work for the UAV world-model arena ([protocol](../uav_arena/README.md)), grouped by how each
work relates to it. BibTeX keys refer to [uav-arena-related-work.bib](uav-arena-related-work.bib).

**Checked 2026-10-26.** Every arXiv title and author list comes from the arXiv API. Every other
entry was checked on its venue page, handle or DOI. A venue is listed only where the arXiv record
or the venue page states it. Preprints in this area move fast, so search again before submission.
Section 8 was added on 2026-10-27 under the same rules; its venues come from the arXiv records,
Crossref (COMA), PMLR (Jaques et al.) and OpenReview (ME-TRPO).

**Position.** We found no work that evaluates learned world models in a physically coupled
multi-agent system, and none that tests whether cross-agent counterfactual error predicts
closed-loop outcomes. The nearest neighbours:
- **platform:** World-in-World (single agent, video models);
- **model:** SkyJEPA (one quadrotor);
- **task:** CrazyMARL (model-free) and Wahba et al. and Sun et al. (model-based).

## 1. Base models and nearest neighbours

| Work | Venue | Relation to the arena |
|---|---|---|
| [LeWorldModel](https://arxiv.org/abs/2603.19312) `maes2026leworldmodel` | arXiv 2026 | Base JEPA world model (SIGReg); our entity-token models follow it |
| [stable-worldmodel](https://arxiv.org/abs/2605.21800) `maes2026stableworldmodel` | arXiv 2026 | Platform whose MPPI and Lagrangian solvers the arena reuses; reports that prediction error tracks planning success poorly |
| [SkyJEPA](https://arxiv.org/abs/2606.23444) `rao2026skyjepa` | arXiv 2026, under review | Nearest model: JEPA world model with MPPI on one quadrotor; no team, cables or payload |

## 2. Prediction fidelity versus decision quality

All four are single-agent. The cross-agent version is the arena's open question.

| Work | Venue | Relation to the arena |
|---|---|---|
| [Objective Mismatch in Model-based RL](https://arxiv.org/abs/2002.04523) `lambert2020objective` | L4DC 2020 | Ancestor: model likelihood does not track control performance |
| [Prediction Fidelity Is Not Decision Fidelity](https://neurips.cc/virtual/2026/166125) `liu2026predictionfidelity` | NeurIPS 2026 workshop poster | Counterfactual action-ranking diagnostic for robot world models |
| [Decision-Level Auditing of Latent Robot World Models](https://neurips.cc/virtual/2026/165855) `rana2026auditing` | NeurIPS 2026 workshop poster | Audits latent action choices against simulator-realized endpoints on Push-T |
| [AD-WM](https://arxiv.org/abs/2609.30264) `qiu2026adwm` | arXiv 2026 | Action-discriminative world models; elite regret tracks closed-loop success better than factual error. Our layer 3 compares against this |
| [Counterfactual action evaluation in JEPA world models](https://arxiv.org/abs/2610.02860) `subramanian2026counterfactual` | NeurIPS 2026 workshop | Counterfactual evaluation of JEPA predictors on a single agent |

## 3. World-model evaluation platforms and benchmark design

| Work | Venue | Relation to the arena |
|---|---|---|
| [World-in-World](https://arxiv.org/abs/2510.18135) `zhang2026worldinworld` | ICLR 2026 oral | Nearest platform: closed-loop evaluation of world models with one shared planner. Design ideas below |
| [Do World Models Make Better Robots?](https://arxiv.org/abs/2609.29669) `jena2026survey` | arXiv 2026 | Survey of 160 benchmarks: counterfactual ability is rarely tested |
| [How Should World Models Be Evaluated for Embodied Decision-Making?](https://arxiv.org/abs/2606.15032) `yu2026position` | arXiv 2026 | Position paper arguing for decision-centric, layered evaluation |
| [MotionScape](https://arxiv.org/abs/2604.07991) `guo2026motionscape`, [AeroVerse](https://arxiv.org/abs/2408.15511) `yao2024aeroverse` | arXiv | UAV world-model benchmarks for video generation or multimodal agents, not dynamics for control |
| [Independent learning in SMAC](https://arxiv.org/abs/2011.09533) `dewitt2020independent`, [SMACv2](https://arxiv.org/abs/2212.07489) `ellis2022smacv2` | arXiv | Cooperative benchmarks can be solvable without coordination; the arena must show that cross-agent modelling matters |

### Design ideas from World-in-World

> **Its content is unrelated to ours.** It evaluates video-generation world models on single-agent,
> perception-heavy tasks (Habitat navigation, recognition, embodied QA, RLBench manipulation).
> We borrow evaluation design, not results.

1. **One planning loop for every model.** A proposal policy, world-model rollouts of each
   candidate and a revision step, executing only the first part. This is the arena's rule too:
   every model-based participant uses the shared MPPI and only the predictor changes.
2. **A standardized action API.** One controller maps planner actions into each model's input
   format. The arena should name this layer explicitly: setpoint segments for the planners, rotor
   commands for CrazyMARL, and the native action spaces of TD-MPC2 and DreamerV3.
3. **Training scenes disjoint from evaluation scenes.** The arena already keeps disjoint seed
   ranges for dev, test, CrazyMARL training and world-model data.
4. **Controllability beats visual quality.** Action-conditioned prediction accuracy predicted task
   success better than image quality. This is the single-agent precursor of our question; the
   cross-agent counterfactual layer refines it for coupled teams.
5. **A data-scaling axis** (400 to 80K post-training samples). Keep two or three data budgets per
   world-model family.
6. **An inference-compute axis** (world-model calls per episode). Vary the MPPI sample budget to
   separate model error from search limits.
7. **Policies without a world model as baselines.** CrazyMARL plays this role in the arena.

Where to do better: World-in-World reports no variance or confidence intervals, and some reported
gaps are a single episode out of 200. It also has no ground-truth simulator. The arena should
report paired confidence intervals and use its true simulator for exact counterfactual and
decision-level references.

## 4. Multi-agent world models and counterfactuals

| Work | Venue | Relation to the arena |
|---|---|---|
| [MARIE](https://arxiv.org/abs/2406.15836) `zhang2024marie` | Accepted by TMLR | Multi-agent world model for sample-efficient MARL; no counterfactual evaluation |
| [MATWM](https://arxiv.org/abs/2506.18537) `deihim2025matwm` | arXiv 2025 | Transformer world model for MARL sample efficiency |
| [SeqWM](https://arxiv.org/abs/2509.13095) `zhao2025seqwm` | arXiv 2025 | Sequential world models for multi-robot cooperation |
| [COMBO](https://arxiv.org/abs/2404.10775) `zhang2025combo` | ICLR 2025 | Compositional joint-action world model, judged by task success |
| [MACD](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p281.pdf) `chai2024macd` | AAMAS 2024 | World-model counterfactual imagination for credit assignment (used for training, not evaluation) |
| [Counterfactual semantic-social world models](https://arxiv.org/abs/2610.07704) `martinez2026counterfactual` | arXiv 2026 | Counterfactual rollouts used to train independent MARL agents |
| [CounterScene](https://arxiv.org/abs/2603.21104) `jing2026counterscene` | arXiv 2026 | Counterfactual causal reasoning in generative world models for closed-loop driving evaluation |

## 5. The task: multi-UAV cable-suspended transport

| Work | Venue | Relation to the arena |
|---|---|---|
| [CrazyMARL](https://arxiv.org/abs/2509.14126) `lorentz2026crazymarl` | ICRA 2026 | External model-free baseline: decentralized motor-level IPPO, retrained on arena tasks |
| [Kinodynamic planning for a cable-suspended payload](https://arxiv.org/abs/2310.03394) `wahba2024kinodynamic` | IROS 2024 | Published classical pipeline for the static condition ([code](https://github.com/IMRCLab/coltrans-planning) `imrclab2024coltrans`); set aside for now |
| [Cable force allocation for geometric control](https://arxiv.org/abs/2304.02359) `wahba2024cableforce` | RA-L 2024 | Onboard payload controller (`controller_lee_payload.c` in the pinned firmware) |
| [pc-dbCBS](https://arxiv.org/abs/2505.10355) `wahba2025pcdbcbs` | arXiv 2025 | Faster successor planner for physically coupled teams |
| [Agile and cooperative aerial manipulation](https://arxiv.org/abs/2501.18802) `sun2025agile` | Science Robotics 2025 | Strongest model-based counter-argument to "why learn": real-time, robust to load uncertainty and wind; [flight data](https://doi.org/10.5061/dryad.n2z34tn6w) `sun2025dryad` |
| [NMPC for full-pose manipulation](https://arxiv.org/abs/2301.08545) `sun2023nmpc` | arXiv 2023 | Centralized NMPC for a cable-suspended load |
| [NMPC for cooperative transport](https://arxiv.org/abs/2303.06165) `li2023nmpc` | IROS 2023 | NMPC for multi-quadrotor cable-suspended transport |
| [RotorTM](https://arxiv.org/abs/2205.05140) `li2024rotortm` | T-RO 2024 | Open simulator for aerial transport, with taut and slack cables |
| [Adaptive mass estimation for payload transport](https://arxiv.org/abs/2607.00024) `hajieghrary2026adaptive` | IROS 2026 (accepted) | Precedent for the planned physics row with online mass estimation |
| [CBF-based RL for multi-drone transport](https://arxiv.org/abs/2607.20665) `choi2026safe` | RA-L 2026 (early access) | Learned graph-CBF transport with obstacles, planar |
| [MARL for a cable-suspended load](https://arxiv.org/abs/2508.01522) `zeng2025decentralized` | CoRL 2025 | MARL transport that survives losing one aerial vehicle |
| [Imitation of a kinodynamic planner](https://arxiv.org/abs/2510.17143) `agarwal2025decentralized` | MRS 2025 (accepted) | Decentralized real-time planning by imitation; no obstacles |
| [Auto-Multilift](https://arxiv.org/abs/2406.04858) `wang2024automultilift` | arXiv 2024 | Distributed learning and control for cooperative load transport |
| [Geometric control with a cable-suspended rigid body](https://arxiv.org/abs/1403.3684) `lee2014geometric` | arXiv 2014 | Classical multi-quadrotor payload dynamics and control |
| [Neural-Swarm](https://arxiv.org/abs/2003.02992) `shi2020neuralswarm`, [Neural-Swarm2](https://arxiv.org/abs/2012.05457) `shi2021neuralswarm2`, [Neural-Fly](https://arxiv.org/abs/2205.06908) `oconnell2022neuralfly` | ICRA 2020, T-RO 2021, Science Robotics 2022 | Learned interaction and aerodynamics models used for control, not as planning world models |
| [RoVerFly](https://arxiv.org/abs/2509.11149) `kim2025roverfly` | arXiv 2025 | Robust control of a single quadrotor with a suspended payload |
| [OmniDrones](https://arxiv.org/abs/2309.12825) `xu2023omnidrones` | arXiv 2023 | RL platform whose transport tasks use rigid links, not cables |
| [Cooperative payload transportation with PETS](https://hdl.handle.net/10919/120971) `khursheed2024cooperative` | Virginia Tech MS thesis 2024 | PETS world model with MPC for multi-UAV payload transport |

## 6. External baselines and latency

| Work | Venue | Relation to the arena |
|---|---|---|
| [TD-MPC2](https://arxiv.org/abs/2310.16828) `hansen2024tdmpc2` | ICLR 2024 | External model-based RL baseline (official code, trained online) |
| [DreamerV3](https://arxiv.org/abs/2301.04104) `hafner2023dreamerv3` | arXiv 2023 | External model-based RL baseline if time allows |
| [PETS](https://arxiv.org/abs/1805.12114) `chua2018pets` | NeurIPS 2018 | Classic probabilistic-ensemble dynamics model for planning; planned participant |
| [RL from delayed observations via world models](https://arxiv.org/abs/2403.12309) `karamzade2024delayed` | arXiv 2024 | World models that bridge observation delay, related to the latency factor |
| [FlightBench](https://arxiv.org/abs/2406.05687) `yu2024flightbench` | arXiv 2024 | Quadrotor navigation benchmark; latency randomization in training |

## 7. Platform and software

| Work | Use in the arena |
|---|---|
| [MuJoCo](https://doi.org/10.1109/IROS.2012.6386109) `todorov2012mujoco` | Physics engine (MJX for CrazyMARL training) |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `menagerie2022github` | Official Crazyflie 2 airframe |
| [Crazyflie firmware, IMRCLab fork](https://github.com/IMRCLab/crazyflie-firmware/tree/b08d9f3cbcc581d8e7f988cabad22c26d35f4a58) `imrclab2024firmware` | Pinned `controller_lee.c` running onboard every aircraft |

## 8. Evaluation design: layers 2–3 and the headline analysis

No work scores cross-agent counterfactuals of learned world models in a physically coupled team (see
Position), so evaluation layers 2–3 and the headline analysis borrow established practice from
neighbouring fields. Each row says what the arena takes from the work. A section number after a key
marks a work already listed in that section.

| Informs | Work | Venue | What the arena takes from it |
|---|---|---|---|
| Scored states | [Agnostic system identification for model-based RL](https://arxiv.org/abs/1203.1007) `ross2012agnostic` | ICML 2012 | A model has to be accurate on the states the planner visits, so layers 2–3 score states from closed-loop task flights rather than only the training distribution |
| | [Objective Mismatch in Model-based RL](https://arxiv.org/abs/2002.04523) `lambert2020objective` (§2) | L4DC 2020 | Likelihood on the training distribution does not track control performance |
| | [MBPO](https://arxiv.org/abs/1906.08253) `janner2019mbpo`, [ME-TRPO](https://arxiv.org/abs/1802.10592) `kurutach2018metrpo` | NeurIPS 2019, ICLR 2018 | Model error grows away from the data and planners exploit it, which motivates scoring the candidates the planner itself proposes |
| Layer 2 | [Counterfactual multi-agent policy gradients (COMA)](https://arxiv.org/abs/1705.08926) `foerster2018coma` | AAAI 2018 | The standard multi-agent counterfactual: hold the other agents' actions fixed and change one. Layer 2 applies it to predictions instead of credit assignment |
| | [CoPhy](https://arxiv.org/abs/1909.12000) `baradel2020cophy` | ICLR 2020 (spotlight) | Counterfactual physical prediction scored against simulator ground truth, for single scenes without control |
| | [Causal influence detection](https://arxiv.org/abs/2106.03443) `seitzer2021causal` | NeurIPS 2021 | An action's influence on other entities depends on the state, which supports reporting effect sizes per state and a resolution threshold for relative errors |
| | [Social influence as intrinsic motivation](https://arxiv.org/abs/1810.08647) `jaques2019social` | ICML 2019 | One agent's counterfactual actions measure its influence on the others, there on their policies rather than on physics |
| | [Counterfactual action evaluation in JEPA world models](https://arxiv.org/abs/2610.02860) `subramanian2026counterfactual` (§2) | NeurIPS 2026 workshop | Predictors can under-use the action pathway, which motivates the changed aircraft's own error as a control for the cross-agent errors |
| Layer 3 | [Benchmarks for deep off-policy evaluation (DOPE)](https://arxiv.org/abs/2103.16596) `fu2021dope` | ICLR 2021 | Standard metrics for whether an estimator ranks candidates correctly: absolute error, Spearman rank correlation and regret@k |
| | [Contrastive structured world models (C-SWM)](https://arxiv.org/abs/1911.12247) `kipf2020cswm` | ICLR 2020 | Ranking metrics (Hits@1, MRR) as a world-model evaluation |
| | [The value equivalence principle](https://arxiv.org/abs/2011.03506) `grimm2020value` | NeurIPS 2020 | A model only needs to be right on decision-relevant quantities, which grounds scoring models by ranking and regret |
| | [AD-WM](https://arxiv.org/abs/2609.30264) `qiu2026adwm` (§2) | arXiv 2026 | Elite regret tracks closed-loop success better than factual error, for a single agent |
| Headline analysis | [Deep RL at the edge of the statistical precipice](https://arxiv.org/abs/2108.13264) `agarwal2021rliable` | NeurIPS 2021 (outstanding paper) | Interval estimates with few runs: stratified bootstrap confidence intervals and the interquartile mean |
| | [Deep RL that matters](https://arxiv.org/abs/1709.06560) `henderson2018matters` | AAAI 2018 | Seed variance and reporting pitfalls |
| | [How many random seeds?](https://arxiv.org/abs/1806.08295) `colas2018seeds` | arXiv 2018 | Power analysis for the number of seeds; here, the number of models |
| | [World-in-World](https://arxiv.org/abs/2510.18135) `zhang2026worldinworld` (§3) | ICLR 2026 oral | The closest platform precedent for asking which model property predicts closed-loop success; it reports no confidence intervals |
| Stochastic predictors | [PEGASUS](https://arxiv.org/abs/1301.3878) `ng2000pegasus` | UAI 2000 | Fixing the random numbers makes comparisons of stochastic systems deterministic, which supports using the same draws for a plan and its counterfactual |
| | [PETS](https://arxiv.org/abs/1805.12114) `chua2018pets` (§6) | NeurIPS 2018 | The trajectory-sampling propagation that the PETS row adapts |
