# 逐段前后对照

共 45 处正文/图注修改。每项可展开查看完整原文和改文；LaTeX 命令保留，便于核对引用与符号。所有 TODO、表格和公式保持原样。

| 章节 | 修改项 |
|---|---|
| Abstract | [1](#edit-01) |
| Introduction | [2](#edit-02), [3](#edit-03), [4](#edit-04), [5](#edit-05), [6](#edit-06), [7](#edit-07), [8](#edit-08), [9](#edit-09) |
| Related Work | [10](#edit-10), [11](#edit-11), [12](#edit-12) |
| The Arena | [13](#edit-13), [14](#edit-14), [15](#edit-15), [16](#edit-16), [17](#edit-17), [18](#edit-18), [19](#edit-19), [20](#edit-20), [21](#edit-21), [22](#edit-22), [23](#edit-23), [24](#edit-24), [25](#edit-25) |
| Evaluation Layers | [26](#edit-26), [27](#edit-27), [28](#edit-28), [29](#edit-29), [30](#edit-30), [31](#edit-31), [32](#edit-32), [33](#edit-33), [34](#edit-34), [35](#edit-35), [36](#edit-36) |
| Results | [37](#edit-37), [38](#edit-38), [39](#edit-39), [40](#edit-40), [41](#edit-41) |
| Discussion | [42](#edit-42) |
| Fig. 1 图注 | [43](#edit-43) |
| Fig. 5 图注 | [44](#edit-44) |
| Fig. 6 图注 | [45](#edit-45) |

<a id="edit-01"></a>
## 01. Abstract

压缩背景铺垫，集中报告平台、主检验及证据范围。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
World models, learned dynamics models, increasingly sit inside the planners of aerial robots, raising a practical question: does a model that predicts well also fly well? The answer is usually assumed: world models are chosen by their error on held-out data, while a planner uses them on the states it visits, to compare plans. When several aircraft carry one slung load, comparing plans means predicting how one aircraft's move propagates through the cables to the load and the other aircraft. We introduce an open-source arena that tests world models in the loop: three Crazyflie~2 quadrotors carry a load of unknown mass past static and moving obstacles, with delayed observations. A ground-truth simulator and one shared planner, in which only the world model changes, score every model on four layers: open-loop prediction, cross-agent counterfactual prediction, decision quality and mission success. Before any test run, we asked of 30 learned world models whether the cross-agent layer explains success differences that open-loop prediction misses. Open-loop error on the planner's states is the best single guide to success; the cross-agent layer explains part of what it misses, mainly through two of five model families; decision quality tracks success least. Physics that estimates the load mass online flies as well as an oracle, and the best learned models about as often as physics with a nominal mass. World models for a team should therefore be validated in the loop, on the plans the planner flies and on how the team responds to one aircraft's move.
```

**修改后**

```latex
World models are often selected by held-out prediction error, but planners use them to compare plans on the states they visit. For aircraft carrying a shared load, this requires predicting how one aircraft's move affects the load and its teammates. We introduce an open-source arena in which three Crazyflie~2 quadrotors transport a load of unknown mass past static and moving obstacles under delayed observations. A shared planner evaluates world models on four layers: open-loop prediction, cross-agent counterfactual prediction, decision quality and mission success, with exact simulator references for the first three. A pre-registered analysis across 30 trained model instances tests whether cross-agent error adds information about success beyond open-loop error. Open-loop error on the planner's states is the strongest single correlate of success; cross-agent error adds information, mainly through the contrast between PETS and JEPA \texttt{flat}. Decision quality correlates with success least. Physics with online mass estimation achieves a success rate similar to the oracle's, while the best learned models achieve rates similar to nominal-mass physics. These findings support evaluating team world models through complementary prediction diagnostics and closed-loop missions.
```

</details>

<a id="edit-02"></a>
## 02. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Learned dynamics models are entering the planners and controllers of aerial robots. They capture effects that are hard to model from first principles, such as aerodynamic disturbances~\cite{oconnell2022neuralfly} and the downwash between nearby multirotors~\cite{shi2020neuralswarm,shi2021neuralswarm2}; long-horizon world models have been learned for quadrotor control~\cite{rao2026skyjepa}, and a probabilistic ensemble has guided payload transport by four unmanned aerial vehicles (UAVs) in simulation~\cite{khursheed2024cooperative}. Inside a planner, such a model acts as a world model: given the current state and a candidate plan, it predicts how the vehicles and their load will move, and the planner favors the plans whose predicted futures score best.
```

**修改后**

```latex
Learned dynamics models capture aerodynamic disturbances~\cite{oconnell2022neuralfly} and downwash between multirotors~\cite{shi2020neuralswarm,shi2021neuralswarm2}, support long-horizon quadrotor control~\cite{rao2026skyjepa}, and guide cooperative payload transport in simulation~\cite{khursheed2024cooperative}. Inside a planner, a world model predicts the futures of candidate plans so that the planner can compare their costs.
```

</details>

<a id="edit-03"></a>
## 03. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
In aviation, a controller is not cleared to fly because its model fits data: it is tested in the loop, first in simulation and then with its hardware, before flight tests~\cite{jung2007hil,belcastro2012validation}. Learned world models, by contrast, are usually selected and compared by their open-loop prediction error on held-out data. This is cheap and needs no planner, but held-out data are not the states a planner visits. In our development runs, a multilayer perceptron (MLP) world model erred ten times more on the states the planner visits than on held-out data, and with it the planner completed fewer than one mission in five. The question that matters in flight is a different one: does a model that predicts well also fly well?
```

**修改后**

```latex
Aviation controllers undergo closed-loop testing in simulation and with hardware before flight tests~\cite{jung2007hil,belcastro2012validation}. Learned world models are commonly selected by held-out prediction error, yet the planner may visit different states. In our development runs, a multilayer perceptron (MLP) had ten times its held-out error on the planner's states and completed fewer than one mission in five. Does a model that predicts well also fly well?
```

</details>

<a id="edit-04"></a>
## 04. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Evidence that prediction quality and decision quality can part ways is accumulating. A dynamics model's one-step likelihood need not track the reward its planner achieves~\cite{lambert2020objective}; low prediction error does not guarantee good action choices~\cite{qiu2026adwm,liu2026predictionfidelity}; and in quadrotor simulation, predictors whose errors differ by an order of magnitude reach nearly the same success rate under one planner~\cite{min2026same}. When several aircraft carry one slung load~\cite{lee2014geometric,wahba2024kinodynamic,sun2025agile,lorentz2026crazymarl}, the cables couple them, and a planner that compares candidate plans must predict how one aircraft's move propagates through the cables to the load and to the other aircraft, an effect that the error on a single plan may hide. We therefore measure prediction where the planner uses it and ask two questions. Does a model's error on the plans the planner flies tell whether it flies? And does its error on how one aircraft's move propagates tell more?
```

**修改后**

```latex
Prediction accuracy and control performance can diverge: one-step likelihood need not track reward~\cite{lambert2020objective}, low prediction error can coexist with poor action choices~\cite{qiu2026adwm,liu2026predictionfidelity}, and quadrotor predictors with errors an order of magnitude apart can achieve similar success under one planner~\cite{min2026same}. Cooperative slung-load transport adds physical coupling~\cite{lee2014geometric,wahba2024kinodynamic,sun2025agile,lorentz2026crazymarl}: one aircraft's move affects the load and the other aircraft through their cables. We ask whether predicting these cross-agent effects adds information about mission success beyond trajectory prediction error.
```

</details>

<a id="edit-05"></a>
## 05. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
We study these questions with an arena that tests world models in the loop (Fig.~\ref{fig:loop}). Three Crazyflie~2 quadrotors carry a cable-suspended load of unknown mass past static and moving obstacles (Fig.~\ref{fig:scene}), with observations delayed by up to 200\,ms. A ground-truth simulator supplies exact references, including the true effect of changing one aircraft's plan. The world models under test all plan with one shared sampling-based planner in which only the world model changes, so differences in flight can be traced to the model. Four evaluation layers then say whether a model flies and where it falls short: open-loop prediction and cross-agent counterfactual prediction, which pose the two questions, then decision quality and closed-loop mission success. The analysis was fixed before any test-set run, so it could not be tuned to the test results.
```

**修改后**

```latex
Our arena tests world models in a shared planner (Fig.~\ref{fig:loop}). Three Crazyflie~2 quadrotors carry a load of unknown mass past static and moving obstacles, with observations delayed by up to 200\,ms (Fig.~\ref{fig:scene}). Only the predictor changes within the shared planning protocol. Four layers measure open-loop prediction, cross-agent counterfactual prediction, decision quality and closed-loop mission success. Exact simulator rollouts supply the references for layers~1--3; the analysis was registered before test-set evaluation.
```

</details>

<a id="edit-06"></a>
## 06. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Across 30 learned world models, open-loop error on the plans the planner flies is the best single guide to mission success, and, as the pre-registered test found, cross-agent counterfactual error explains part of what it misses (Fig.~\ref{fig:scatter}). The evidence comes mainly from one contrast: a probabilistic ensemble predicts the flown plans less accurately than a transformer that encodes the whole team as one token per time step, yet predicts better how one aircraft's move spreads through the team, and flies more often. Decision quality, the layer closest to the mission, tracks success least. A physics model that estimates the load mass online flies as well as an oracle that knows the true state and mass, so the arena measures how close learned models come to physics rather than assuming that they must beat it. A world model for a team, we argue, should be judged by the missions it flies.
```

**修改后**

```latex
Across 30 trained model instances, open-loop error on U's flight plans is the strongest single correlate of mission success, while cross-agent error adds information beyond it (Fig.~\ref{fig:scatter}). This result depends mainly on the PETS--JEPA \texttt{flat} contrast: PETS has larger open-loop error but smaller cross-agent error and higher success. Decision quality correlates with success least. Physics with online mass estimation achieves a success rate similar to the oracle's, providing a reference for the learned models.
```

</details>

<a id="edit-07"></a>
## 07. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\item An open-source arena for cooperative slung-load transport with a hidden load mass, observation delay and moving obstacles; a shared planner and predictor interface; and reference rows: an oracle and physics models with a nominal or an online-estimated load mass.
```

**修改后**

```latex
\item An open-source cooperative transport arena with hidden load mass, delayed observations and moving obstacles, a shared planner and predictor interface, and oracle and physics references.
```

</details>

<a id="edit-08"></a>
## 08. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\item A four-layer evaluation that uses the simulator for exact references: open-loop prediction, cross-agent counterfactual prediction, decision quality and closed-loop mission success.
```

**修改后**

```latex
\item A four-layer evaluation linking trajectory prediction, cross-agent effects, decision quality and mission success, with exact simulator references.
```

</details>

<a id="edit-09"></a>
## 09. Introduction

减少重复设问和结果复述；用明确方法名称承接主要比较。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\item A pre-registered test across 30 learned world models, which finds that open-loop error on the planner's states is the best single guide to mission success and that cross-agent counterfactual error explains success beyond it.
```

**修改后**

```latex
\item A pre-registered analysis of the relationship between prediction diagnostics and mission success across 30 trained model instances.
```

</details>

<a id="edit-10"></a>
## 10. Related Work

合并相近文献叙述，保留具体差异和原引用归属。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Assuring learned components in aviation.} Aviation regulators have issued first objectives and a roadmap for assuring machine-learning components~\cite{easa2024concept,faa2024roadmap}. For a learned surrogate model, the European guidance ties the verification of its accuracy to the high-fidelity reference model it approximates~\cite{easa2024concept}. A world model inside a planner acts only through the planner's decisions, so we also test whether its accuracy carries over to them.
```

**修改后**

```latex
\textbf{Assuring learned components in aviation.} Aviation regulators provide objectives and a roadmap for assuring machine-learning components~\cite{easa2024concept,faa2024roadmap}. European guidance ties a learned surrogate's accuracy to its high-fidelity reference model~\cite{easa2024concept}. We also evaluate how that accuracy translates into a planner's decisions.
```

</details>

<a id="edit-11"></a>
## 11. Related Work

合并相近文献叙述，保留具体差异和原引用归属。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Decision-aware evaluation of world models.} Decision-level diagnostics score a world model by how it ranks a fixed bank of candidate actions~\cite{qiu2026adwm,liu2026predictionfidelity,rana2026auditing}, and such diagnostics now cover multi-agent world models~\cite{zou2026mawam}. Yu et~al.~\cite{yu2026position} propose a ladder of evaluation levels and a benchmark protocol with exact simulator resets and a fixed optimizer, and ask that lower-level measures be validated against decisions. We run such a protocol, at the ladder's levels of interventional fidelity and planning utility, for aircraft coupled through a shared load. We add a cross-agent counterfactual layer scored against exact ground truth, and score decisions by the plan the planner would fly rather than by how a model ranks candidates.
```

**修改后**

```latex
\textbf{Decision-aware evaluation of world models.} Decision-level diagnostics assess rankings of fixed candidate actions~\cite{qiu2026adwm,liu2026predictionfidelity,rana2026auditing}, including in multi-agent settings~\cite{zou2026mawam}. Yu et~al.~\cite{yu2026position} propose an evaluation ladder and a protocol with exact simulator resets and a fixed optimizer. Our protocol addresses interventional fidelity and planning utility for aircraft coupled through a shared load: it measures cross-agent counterfactual error against exact rollouts and scores the planner's execution plan rather than candidate rankings.
```

</details>

<a id="edit-12"></a>
## 12. Related Work

合并相近文献叙述，保留具体差异和原引用归属。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Cooperative slung-load transport.} Methods for teams of multirotors carrying cable-suspended loads include geometric control~\cite{lee2014geometric}, kinodynamic planning~\cite{wahba2024kinodynamic}, learned motor-level policies~\cite{lorentz2026crazymarl} and model-based control that copes with an unknown load mass~\cite{sun2025agile,hajieghrary2026adaptive}. Dynamics models have been compared under one planner for a single quadrotor~\cite{min2026same}, for a single UAV carrying a suspended payload~\cite{belkhale2021meta,jin2025neural} and, in simulation, for a team of four~\cite{khursheed2024cooperative}. We compare 30 learned world models for a team of three quadrotors under one planner.
```

**修改后**

```latex
\textbf{Cooperative slung-load transport.} Approaches include geometric control~\cite{lee2014geometric}, kinodynamic planning~\cite{wahba2024kinodynamic}, learned motor-level policies~\cite{lorentz2026crazymarl} and control under unknown load mass~\cite{sun2025agile,hajieghrary2026adaptive}. Dynamics models have been compared under a common planner for a quadrotor~\cite{min2026same}, a UAV carrying a suspended payload~\cite{belkhale2021meta,jin2025neural} and a four-UAV team in simulation~\cite{khursheed2024cooperative}. We connect such controlled comparisons to cross-agent prediction and closed-loop success.
```

</details>

<a id="edit-13"></a>
## 13. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Our arena holds everything but the world model fixed: the aircraft and their onboard control, the missions, the observations and the planner with its cost. This section describes each, then the models under test; Table~\ref{tab:arena} lists the settings.
```

**修改后**

```latex
The world-model comparison fixes the aircraft, onboard control, missions, observations, planner and cost. Table~\ref{tab:arena} summarizes the settings.
```

</details>

<a id="edit-14"></a>
## 14. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Cooperative slung-load transport couples the aircraft through the cables: when one aircraft changes course, the load swings and the other two aircraft feel the pull. Predicting that propagation is what layer~2 probes (Section~\ref{sec:layers}). Three further features make prediction matter: the load mass is unknown, observations arrive late and obstacles move.
```

**修改后**

```latex
Cables couple the aircraft through the shared load; layer~2 measures predictions of this coupling (Section~\ref{sec:layers}). Hidden load mass, delayed observations and moving obstacles make the task sensitive to prediction.
```

</details>

<a id="edit-15"></a>
## 15. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Three Crazyflie~2 quadrotors, with airframes from MuJoCo Menagerie~\cite{todorov2012mujoco,menagerie2022github}, carry one load on tension-only cables. Each aircraft flies with the geometric tracking controller of the Crazyflie firmware~\cite{imrclab2024firmware}, based on Lee et~al.~\cite{lee2010geometric} and compiled unchanged, so a planner commands setpoints rather than thrusts. The load mass changes from layout to layout and is hidden from every decision maker except the oracle (Section~\ref{sec:planner}), so the same setpoints move the team differently from one layout to the next.
```

**修改后**

```latex
Three Crazyflie~2 quadrotors from MuJoCo Menagerie~\cite{todorov2012mujoco,menagerie2022github} carry a load on tension-only cables. Each uses the Crazyflie firmware's geometric tracking controller~\cite{imrclab2024firmware}, based on Lee et~al.~\cite{lee2010geometric} and compiled unchanged. The planner commands setpoints rather than thrusts. Load mass varies by layout and is hidden from every decision maker except the oracle (Section~\ref{sec:planner}).
```

</details>

<a id="edit-16"></a>
## 16. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
A mission starts from a settled hover and succeeds once the load settles at a goal past four pillars and, in half of the conditions, two spheres that shuttle across the path (Fig.~\ref{fig:scene}). It fails when an aircraft, the load or a cable touches an obstacle, when two aircraft collide, when an aircraft or the load hits the floor, when the team leaves the workspace, or when time runs out.
```

**修改后**

```latex
Missions start from settled hover and succeed when the load settles at the goal beyond four pillars, with two spheres crossing the path in half the conditions (Fig.~\ref{fig:scene}). Failure occurs on obstacle contact by an aircraft, load or cable; aircraft--aircraft collision; aircraft or load contact with the floor; workspace exit; or timeout.
```

</details>

<a id="edit-17"></a>
## 17. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Decisions are made ten times a second on a state up to 200\,ms old, while the onboard controllers act on the true current state (Fig.~\ref{fig:decision}). A world model must therefore also predict how the team moved during the delay, from the setpoints sent since. Observations are states rather than images: this isolates the prediction of dynamics, which the layers score, from perception, which we leave to future work (Section~\ref{sec:discussion}).
```

**修改后**

```latex
The planner acts at 10\,Hz on states delayed by up to 200\,ms; onboard Lee controllers use the true current state (Fig.~\ref{fig:decision}). The predictor bridges the delay using previously sent setpoints. Observations contain states rather than images; controller and motor states remain hidden from the planner and predictor.
```

</details>

<a id="edit-18"></a>
## 18. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Two obstacle settings, the pillars alone or with the moving spheres, crossed with four delays give eight conditions. A layout keeps its pillars and load mass in all eight and its sphere motions at every delay, so conditions are compared on the same layouts; the test layouts give 400 test episodes. Training, validation, development and test layouts are disjoint.
```

**修改后**

```latex
Two obstacle settings and four delays give eight conditions (Table~\ref{tab:arena}). Each layout retains its pillars and load mass across conditions and its sphere motions across delays, enabling paired comparisons. Training, validation, development and test layouts are disjoint.
```

</details>

<a id="edit-19"></a>
## 19. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
The reference rows and the learned world models all plan with the same model predictive path integral (MPPI) planner~\cite{williams2017mppi}, as implemented in stable-worldmodel~\cite{maes2026stableworldmodel}, with the same actions and cost; only the predictor changes. At each decision, MPPI samples candidate plans for the next 1.5\,s, has the predictor roll each one out, and commands a cost-weighted average of them (Fig.~\ref{fig:loop}).
```

**修改后**

```latex
Reference predictors and learned world models use the same model predictive path integral (MPPI) planner~\cite{williams2017mppi}, implemented in stable-worldmodel~\cite{maes2026stableworldmodel}, with identical actions and cost. MPPI samples candidate plans, rolls them out with the predictor and commands their cost-weighted average (Fig.~\ref{fig:loop}).
```

</details>

<a id="edit-20"></a>
## 20. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
The predictor bridges the delay itself, may keep a history of the episode's earlier observations, and returns for each candidate what the cost reads (Fig.~\ref{fig:decision}). The cost charges the load's distance from the goal, its speed near the goal and the size of the moves, and penalizes predicted violations of safety margins: clearance from the obstacles for aircraft, load and cables, separation between the aircraft, and the heights of the load above the floor and of each aircraft above the load. For every row alike, the spheres' future positions are extrapolated from their observed velocity, and on a given layout MPPI draws the same random samples.
```

**修改后**

```latex
The predictor compensates for delay, may retain observation history, and returns aircraft positions and load position and velocity for each candidate (Fig.~\ref{fig:decision}). The cost combines load distance to the goal, speed near the goal and move size with safety penalties for obstacle clearance of aircraft, load and cables, aircraft separation, load height above the floor and aircraft height above the load. All predictors share velocity-based extrapolation of sphere positions and the same MPPI random samples on each layout.
```

</details>

<a id="edit-21"></a>
## 21. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
With the true simulator as its predictor, the planner becomes the oracle~U: it starts every rollout from the true, undelayed state, with the true load mass and the onboard states, so its predictions are exact. This privilege makes U the answer key for layers~1--3.
```

**修改后**

```latex
The oracle~U uses the true simulator, initializing rollouts with the current undelayed state, true load mass and onboard states. Its exact predictions provide reference rollouts for all three diagnostic layers.
```

</details>

<a id="edit-22"></a>
## 22. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Reference rows.} Two physics rows set the scale for the learned models. They run U's simulator without its privilege: from the delayed observation, they carry the unobserved controller and motor states in a copy of the known onboard flight stack and bridge the delay by replaying the setpoints already sent, as a Smith predictor does; with the true mass, they reproduce U's predictions. Nominal physics assumes a fixed load mass; physics with online mass estimation keeps 16 models on a 2\,g grid and plans with the one that best predicted the preceding 0.5\,s of observations.
```

**修改后**

```latex
\textbf{Reference rows.} Both physics predictors use the simulator and maintain controller and motor states in a copy of the known flight stack. From delayed observations, they replay sent setpoints to compensate for delay, as a Smith predictor does; given the true mass, they reproduce U's predictions. Nominal physics assumes a fixed mass. Online mass estimation maintains 16 instances with candidate masses from 10 to 40\,g in 2\,g increments, selecting the instance that best predicted the preceding 0.5\,s of observations.
```

</details>

<a id="edit-23"></a>
## 23. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Learned world models.} The learned models differ in how they represent the team. The entity-token JEPA (joint-embedding predictive architecture) follows LeWorldModel~\cite{maes2026leworldmodel}. It encodes recent observations as one token per aircraft and one for the load and, conditioned on each aircraft's setpoints, predicts future tokens in latent space; a readout, trained jointly, decodes them into the positions and velocities the cost reads. Its three variants differ only in how information flows between entities, so they vary the cross-agent structure that layer~2 probes. In \texttt{full}, every token attends to every token; in \texttt{no\_cross}, aircraft tokens do not attend to other aircraft, so information between aircraft passes only through the load token; \texttt{flat} merges each frame into one token. W-ar, an autoregressive state model, uses the \texttt{full} backbone but predicts state changes directly, step by step, without latent targets. PETS (probabilistic ensembles with trajectory sampling)~\cite{chua2018pets} is an ensemble of probabilistic networks, the model that has guided four-UAV transport in simulation~\cite{khursheed2024cooperative}. Because the planner takes one trajectory per candidate, each candidate follows the mean of one randomly drawn member rather than the sampled particles of published PETS.
```

**修改后**

```latex
\textbf{Learned world models.} We use a joint-embedding predictive architecture (JEPA) based on LeWorldModel~\cite{maes2026leworldmodel}. Conditioned on aircraft setpoints, it predicts future latent tokens from recent observations, with one token per aircraft and one for the load. A jointly trained readout decodes the states required by the cost. In \texttt{full}, all tokens attend to one another; in \texttt{no\_cross}, aircraft tokens communicate only through the load token; \texttt{flat} uses one token per frame. W-ar uses the \texttt{full} backbone to predict state changes autoregressively, without latent targets. PETS (probabilistic ensembles with trajectory sampling)~\cite{chua2018pets} uses an ensemble of probabilistic networks, as in four-UAV transport~\cite{khursheed2024cooperative}. To match the planner's single-trajectory interface, each candidate follows the mean of one randomly selected member rather than sampled particles.
```

</details>

<a id="edit-24"></a>
## 24. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Data and training.} The training and validation data come from a behavior policy that flies the team in formation between random waypoints, without obstacles, and perturbs each aircraft's moves at MPPI's sampling scale (Table~\ref{tab:arena}), so that they cover the moves the planner commands. The data behind the MLP of Section~\ref{sec:intro} did not; retrained on these, that MLP is the example row of Table~\ref{tab:results}, outside the 30. For each of the five learned rows, at most eight configurations were compared by validation error alone, the load position error on held-out episodes, and the chosen one was trained with three new seeds on all 2,000 training episodes and on the first 500. Within each model row, both budgets use the same configuration and number of optimization steps. This gives 30 learned world models of similar size (Table~\ref{tab:arena}).
```

**修改后**

```latex
\textbf{Data and training.} A behavior policy collects obstacle-free formation flights between random waypoints, perturbing each aircraft's moves at MPPI's sampling scale. This covers planner-scale actions missing from the development MLP's data; the MLP retrained on these data appears in Table~\ref{tab:results}, outside the primary model set. For each of five predictor variants (JEPA \texttt{full}, \texttt{no\_cross}, \texttt{flat}, W-ar and PETS), up to eight configurations were compared using held-out load-position error alone. The selected configuration was trained with three new seeds on 2,000 episodes and the first 500 of the same dataset. Both budgets use the same configuration and number of optimization steps within each variant, yielding $5\times2\times3=30$ trained model instances of similar size (Table~\ref{tab:arena}).
```

</details>

<a id="edit-25"></a>
## 25. The Arena

复用表内设置，明确控制接口、信息权限和模型实例；保留比较协议。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Policies without the shared planner.} Three policies show how far published methods get without the shared planner; each was trained with its official code on arena layouts, with three seeds (Table~\ref{tab:arena}). CrazyMARL~\cite{lorentz2026crazymarl} commands rotor thrusts from each aircraft's fresh own state and delayed observations of the rest; it was retrained for its official budget of $2\times10^9$ steps per seed, with 8,192 parallel environments (half the official 16,384), in its own simulator, configured with the arena's airframes, cables, obstacles and delays and the nominal load mass. TD-MPC2~\cite{hansen2024tdmpc2} and DreamerV3~\cite{hafner2023dreamerv3} trained online for $2\times10^6$ steps per seed from the delayed observation with the state-based settings of their official code, on about twelve times the world models' data, with a reward for the load's progress, a bonus on success and a penalty on failure. The policies do not implement the predictor interface that layers~1--3 call, so they are scored on layer~4 only.
```

**修改后**

```latex
\textbf{Policies without the shared planner.} Three policies were trained with official code and three seeds on arena layouts. CrazyMARL~\cite{lorentz2026crazymarl} commands rotor thrusts using each aircraft's current own state and delayed observations of the rest. It uses its own simulator with the arena's airframes, cables, obstacles and delays, nominal load mass, and the official budget of $2\times10^9$ steps per seed with 8,192 parallel environments (official: 16,384). TD-MPC2~\cite{hansen2024tdmpc2} and DreamerV3~\cite{hafner2023dreamerv3} use their official state-based settings and delayed observations for $2\times10^6$ online steps per seed, about twelve times the world models' data. Their reward combines load progress, a success bonus and a failure penalty. These policies lack the predictor interface and are evaluated on layer~4 only.
```

</details>

<a id="edit-26"></a>
## 26. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Layer~4, mission success, says whether a model flies; layers~1--3 locate where it falls short, each closer to the mission than the last (Fig.~\ref{fig:loop}). Unlike the validation error of Section~\ref{sec:rows}, layers~1--3 ask their questions on the states and plans of flight, the same for every model: at states that U planned from in its test flights, which the planner reaches when its predictions are exact, the model predicts as it would inside the planner. The true simulator, restarted from each state with its load mass and onboard states, gives the exact answers (Fig.~\ref{fig:probes}).
```

**修改后**

```latex
Layers~1--3 diagnose predictions and decisions at the same states and plans from U's test flights, rather than on held-out validation episodes. Each model receives the observations available to it inside the planner; exact simulator resets restore the true mass and onboard states to obtain reference rollouts (Fig.~\ref{fig:probes}). Layer~4 measures success along each model's own closed-loop trajectories.
```

</details>

<a id="edit-27"></a>
## 27. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Let $x_{i,k}(a)$ be the true position of part $i$ of the team, the load or an aircraft, at the end of decision $k$ of a plan $a$, from an exact rollout, and $\hat{x}_{i,k}(a)$ a model's prediction of it.
```

**修改后**

```latex
Let $x_{i,k}(a)$ denote the true position of the load or aircraft $i$ after decision $k$ of plan $a$, and $\hat{x}_{i,k}(a)$ its model prediction.
```

</details>

<a id="edit-28"></a>
## 28. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Layer 1: open-loop prediction.} Does the model predict where the planner's plan takes the load? At each question, it rolls out U's plan $a$, and $L_1$ averages its error $\|\hat{x}_{i,k}(a) - x_{i,k}(a)\|$ on the load over the plan's decisions and the questions. Layer~1 catches a model with a low validation error but a large error on the states and plans of flight, like the MLP of Section~\ref{sec:intro}.
```

**修改后**

```latex
\textbf{Layer 1: open-loop prediction.} The model rolls out U's plan $a$. $L_1$ averages the load-position error $\|\hat{x}_{i,k}(a)-x_{i,k}(a)\|$ over the plan's decisions and the probe states.
```

</details>

<a id="edit-29"></a>
## 29. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Layer 2: cross-agent counterfactual prediction.} Does the model predict how a change in one aircraft's plan propagates through the cables to the load and the other aircraft? MPPI's candidates differ by such changes, on every aircraft at once, and the planner weighs them by the costs the model predicts for them. Each aircraft in turn has its part of $a$ perturbed eight times by Gaussian noise at MPPI's sampling scale, 0.04\,m per axis and decision, and for a changed plan $a'$ the error on part $i$ compares the predicted and true effects of the change:
```

**修改后**

```latex
\textbf{Layer 2: cross-agent counterfactual prediction.} We perturb one aircraft's part of $a$ at a time to measure the predicted response of the load and teammates. Each aircraft receives eight Gaussian perturbations at MPPI's sampling scale, 0.04\,m per axis and decision. For a perturbed plan $a'$, the effect-prediction error on part $i$ is
```

</details>

<a id="edit-30"></a>
## 30. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
where $\Delta x_{i,k} = x_{i,k}(a') - x_{i,k}(a)$ is the true effect and $\Delta\hat{x}_{i,k}$ the predicted one. An error that a model makes with and without the change cancels, so layer~2 scores the predicted effect alone. $L_2$ averages, over the changes and the questions, half the error on the load plus half the mean error on the two unchanged aircraft. On U's development states, the median true effects are 6.2\,cm on the load and 2.7\,cm on another aircraft.
```

**修改后**

```latex
where $\Delta x_{i,k}=x_{i,k}(a')-x_{i,k}(a)$ is the true effect and $\Delta\hat{x}_{i,k}$ its prediction. Errors shared by the original and perturbed predictions cancel in this difference. $L_2$ averages half the load error plus half the mean error on the two unchanged aircraft, over perturbations and probe states. Median true effects on U's development states are 6.2\,cm on the load and 2.7\,cm on another aircraft.
```

</details>

<a id="edit-31"></a>
## 31. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Layer 3: decision quality.} Do the model's predictions lead the planner to a costlier plan than exact predictions would? The model predicts the candidates that U's MPPI scored at the question, and the shared cost and the frozen MPPI update turn its predictions into the plan the planner would fly. Its regret is that plan's true cost minus that of U's executed plan, and the decision is costly when the regret exceeds one cost unit, approximately a 1.7\,cm reduction in a safety margin or 0.6\,m less progress; layer~3 thus weighs an error by what it costs rather than by its size in centimeters~\cite{grimm2020value}. $L_3$ is the share of costly decisions, counted on the questions at which U's own plan, a weighted mean of its candidates, is within one cost unit of the best of them: 89\% of questions, the same for every model.
```

**修改后**

```latex
\textbf{Layer 3: decision quality.} Each model predicts the same candidates scored by U's MPPI. The shared cost and frozen MPPI update produce a cost-weighted execution plan. Regret is its true cost minus that of U's execution plan; a decision is costly if regret exceeds one cost unit, approximately a 1.7\,cm loss of safety margin or 0.6\,m less progress. This measures an error's decision cost rather than its magnitude~\cite{grimm2020value}. $L_3$ is the costly-decision rate on probe states where U's own weighted plan is within one cost unit of its best candidate: 89\% of states, identical across models.
```

</details>

<a id="edit-32"></a>
## 32. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Layer 4: mission success.} Does the model fly? A world model flies the test episodes once inside the shared planner, and a policy once on its own; $J$ is the success rate. Layer~4 alone sees the states that the model's own flight reaches, and the errors that compound along it.
```

**修改后**

```latex
\textbf{Layer 4: mission success.} Each world model runs the test episodes in the shared planner, and each policy runs them independently. $J$ is the success rate. This layer captures errors accumulating along each model's own flight.
```

</details>

<a id="edit-33"></a>
## 33. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Before any test-set run, an analysis plan, released with the arena's code under a version tag, fixed one question and the rule for answering it: across the 30 learned world models, does layer~2 explain differences in $J$ that layer~1 misses? The primary statistic is the partial Spearman correlation between $L_2$ and $J$ given $L_1$, computed from the pairwise Spearman correlations $\rho$ across the 30 models:
```

**修改后**

```latex
The version-tagged analysis plan, fixed before test-set evaluation, tests whether $L_2$ adds information about $J$ beyond $L_1$ across the 30 model instances. The primary statistic is the partial Spearman correlation, computed from pairwise rank correlations $\rho$:
```

</details>

<a id="edit-34"></a>
## 34. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Rank correlations need only a monotone relation, and the partial correlation asks what layer~2 adds once layer~1 is known. Its 95\% interval resamples the test layouts, each with its episodes and questions, and recomputes every model on each resample, so that models and layers stay paired; the 30 models stay fixed (Table~\ref{tab:arena}).
```

**修改后**

```latex
The 95\% interval resamples test layouts with their episodes and probe states, recomputing all models on each resample to preserve pairing across models and layers. The 30 model instances remain fixed (Table~\ref{tab:arena}).
```

</details>

<a id="edit-35"></a>
## 35. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
An interval entirely below zero supports the claim, with higher cross-agent error going with lower success; an interval that contains zero leaves it not established. Every model that completes the protocol enters the analysis; all 30 did. Simulated power is 67--79\% for a partial correlation of 0.43--0.48.
```

**修改后**

```latex
An interval entirely below zero supports the claim; one containing zero leaves it unestablished. All 30 models completed the protocol and enter the analysis. Simulated power is 67--79\% for a partial correlation of 0.43--0.48.
```

</details>

<a id="edit-36"></a>
## 36. Evaluation Layers

去除导读问句，直接定义操作与统计；公式和筛选规则不变。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
Secondary analyses carry intervals but no decision rule: layer~3 analyzed like layer~2, the plain correlation of each layer with $J$, the primary statistic per condition, and the comparisons of the two data sizes and of the three JEPA variants, the latter paired by seed and data size. The released code and analysis plan give the remaining settings, such as the cost weights and the scoring of stochastic predictors, and the choices made on development data.
```

**修改后**

```latex
Secondary analyses report intervals without a decision rule: layer~3 conditioned on layer~1, each layer's plain correlation with $J$, the primary statistic by condition, data-budget comparisons and JEPA-variant comparisons paired by seed and budget. The released analysis plan and code specify cost weights, stochastic-predictor scoring and development-set choices.
```

</details>

<a id="edit-37"></a>
## 37. Results

先给发现及证据，避免再次提出问题；将机制讨论集中到 Discussion。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{The reference rows set the scale, and the best learned world models fly about as often as nominal physics.} U predicts exactly, so its failures, nearly all of them timeouts, belong to the planner's search rather than to a model (Table~\ref{tab:results} and Fig.~\ref{fig:outcomes}). Physics with online mass estimation is level with U. With the nominal load mass, the same physics succeeds in about half of the missions, failing most often by collisions between aircraft, mostly on the heaviest loads, which pull the cables inward harder than it predicts. With all the training data, JEPA \texttt{full} and \texttt{no\_cross} and W-ar fly about as often as nominal physics, while PETS and \texttt{flat} succeed in at most one mission in four; unlike U, the learned rows fail mostly by collisions (Fig.~\ref{fig:outcomes}). Of the policies trained without the shared planner, CrazyMARL succeeds in about two missions out of five, and TD-MPC2 and DreamerV3 in none.
```

**修改后**

```latex
\textbf{The best learned world models approach nominal physics in mission success.} Oracle U's failures are nearly all timeouts despite exact predictions (Table~\ref{tab:results}; Fig.~\ref{fig:outcomes}). Physics with online mass estimation achieves a similar success rate. Nominal-mass physics succeeds in about half the missions, with aircraft--aircraft collisions its most frequent failure, concentrated on heavy loads. At 2,000 training episodes, JEPA \texttt{full}, \texttt{no\_cross} and W-ar achieve success rates similar to nominal physics; PETS and \texttt{flat} succeed in at most a quarter of missions. Learned-model failures are mostly collisions. Among independent policies, CrazyMARL succeeds in about two missions in five; TD-MPC2 and DreamerV3 have no successes.
```

</details>

<a id="edit-38"></a>
## 38. Results

先给发现及证据，避免再次提出问题；将机制讨论集中到 Discussion。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Cross-agent prediction explains success that open-loop prediction misses.} Open-loop error on U's plans is the best single guide to success (Fig.~\ref{fig:scatter}, left), but it misranks some models. PETS predicts the plans less accurately than JEPA \texttt{flat}, yet flies more often (23\% against 16\% at 2,000 episodes), and it predicts better how a change in one aircraft's plan spreads to the rest of the team (Fig.~\ref{fig:scatter}, middle). The pre-registered test asks whether, across the 30 models, cross-agent error explains success that open-loop error misses, and it does: once open-loop error is accounted for, higher cross-agent error goes with lower success, with a partial rank correlation of $-0.39$ (95\% interval $-0.60$ to $-0.27$), so the claim is supported. The evidence rests on these two rows: with either left out, a check outside the analysis plan, the interval contains zero (Fig.~\ref{fig:primary}). Open-loop error alone can therefore put a team's world models in the wrong order.
```

**修改后**

```latex
\textbf{Cross-agent error adds information about mission success beyond open-loop error.} Open-loop error on U's plans is the strongest single correlate of success (Fig.~\ref{fig:scatter}). Controlling for it, cross-agent error remains negatively associated with success across the 30 model instances (partial Spearman correlation $-0.39$, 95\% interval $[-0.60,-0.27]$), supporting the pre-registered claim. PETS illustrates the additional information: it has larger open-loop error than JEPA \texttt{flat}, but smaller cross-agent error and higher success (23\% versus 16\% at 2,000 episodes). In an exploratory check, removing either row makes the interval include zero (Fig.~\ref{fig:primary}), indicating that this contrast is central to the pooled result.
```

</details>

<a id="edit-39"></a>
## 39. Results

先给发现及证据，避免再次提出问题；将机制讨论集中到 Discussion。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Layer~2's contribution is clearest with the pillars alone and short delays.} Layer~2's partial correlation is negative in all eight conditions, but its interval lies below zero only with the pillars alone and delays up to 100\,ms (Fig.~\ref{fig:primary}). Layer~1 stays as closely tied to success in every condition as in the pooled test.
```

**修改后**

```latex
\textbf{The clearest layer~2 associations occur with static obstacles and short delays.} Partial correlations are negative in all eight conditions, with intervals below zero only for pillars alone at delays up to 100\,ms (Fig.~\ref{fig:primary}). Layer~1 remains closely associated with success across conditions.
```

</details>

<a id="edit-40"></a>
## 40. Results

先给发现及证据，避免再次提出问题；将机制讨论集中到 Discussion。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Decision quality, the layer closest to the mission, tracks success least.} Every learned model makes costly decisions at roughly half of the questions, and the share follows success only loosely: with all the training data, PETS and JEPA \texttt{full} make costly decisions about equally often, yet \texttt{full} succeeds more than twice as often (Fig.~\ref{fig:scatter}, right). The partial correlation of layer~3 given layer~1 also lies below zero, but like layer~2's it rests on the PETS and \texttt{flat} rows. A likely reason is that the planner decides again 0.1\,s later, so one costly decision can be undone at the next, while the errors that end a mission compound on the states of the model's own flight, which only layer~4 sees.
```

**修改后**

```latex
\textbf{Decision quality has the weakest correlation with mission success.} Learned models make costly decisions at roughly half the probe states. At 2,000 episodes, PETS and JEPA \texttt{full} have similar costly-decision rates, yet \texttt{full} succeeds more than twice as often (Fig.~\ref{fig:scatter}). Layer~3's partial correlation with success given layer~1 is also negative, with the association again dependent on PETS and \texttt{flat}.
```

</details>

<a id="edit-41"></a>
## 41. Results

先给发现及证据，避免再次提出问题；将机制讨论集中到 Discussion。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{Within JEPA, a token per aircraft matters, and direct attention between aircraft adds little.} The two variants with one token per aircraft and one for the load fly two and a half to three times as often as \texttt{flat}, which merges each frame into a single token, and predict cross-agent effects better (Table~\ref{tab:results}). With all the training data, more than half of \texttt{flat}'s missions end in a collision between aircraft, against fewer than one in five of \texttt{full}'s (Fig.~\ref{fig:outcomes}). Four times the data lifts these two variants from about a third to about half of the missions, while \texttt{flat}'s success does not change detectably. \texttt{no\_cross}, whose aircraft exchange information only through the load token, predicts cross-agent effects slightly better than \texttt{full} and succeeds slightly, but measurably, less often, so this pair runs against the primary relation. The simulator models no aerodynamic interaction between the aircraft: short of contact, they interact only through their cables and the load, the path that \texttt{no\_cross} keeps. More data lowers layers~1 and~2 for every row but raises success only for \texttt{full}, \texttt{no\_cross} and PETS; W-ar, like \texttt{flat}, succeeds about as often with a quarter of the data.
```

**修改后**

```latex
\textbf{Entity tokens improve JEPA performance; direct aircraft attention adds little.} JEPA \texttt{full} and \texttt{no\_cross} achieve two and a half to three times \texttt{flat}'s success and lower cross-agent error (Table~\ref{tab:results}). At 2,000 episodes, aircraft--aircraft collisions end more than half of \texttt{flat}'s missions but fewer than one in five of \texttt{full}'s (Fig.~\ref{fig:outcomes}). Despite slightly lower cross-agent error, \texttt{no\_cross} has slightly but measurably lower success than \texttt{full}.

\textbf{More data improves prediction more consistently than mission success.} Increasing the budget from 500 to 2,000 episodes lowers $L_1$ and $L_2$ for every predictor variant. Success increases for \texttt{full}, \texttt{no\_cross} and PETS; the two entity-token JEPA variants rise from about a third to about half of missions. Changes in \texttt{flat} and W-ar success are not detectable.
```

</details>

<a id="edit-42"></a>
## 42. Discussion

集中解释互补诊断、重规划与耦合边界；所有 TODO 段落原样保留。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\textbf{What each layer tells.} Measured on the states the planner visits, open-loop error is the best single guide to whether a team's world model flies. It scores one plan, while the planner weighs many against one another, and layer~2, which asks how the predictions change when one aircraft's plan does, explains part of what it misses, here mainly the contrast between PETS and \texttt{flat}. Layer~3 scores single decisions at U's states; the next decision can undo a costly one, and the errors that end a mission compound along the model's own flight, which only layer~4 sees. Each layer thus answers a narrower question than mission success, and where two layers disagree, as for PETS and \texttt{flat}, the disagreement points to what a model gets wrong.
```

**修改后**

```latex
\textbf{What each layer tells.} Trajectory and cross-agent errors provide complementary diagnostics, with the latter's additional association here driven mainly by PETS and \texttt{flat}. Neither replaces closed-loop evaluation. Layer~3 measures single decisions at U's states; replanning 0.1\,s later may correct a costly decision, while layer~4 captures errors accumulated on a model's own trajectory. In this arena, aircraft interact only through cables and the load, apart from contact; aerodynamic interaction is absent. That load-mediated path remains available to \texttt{no\_cross}.
```

</details>

<a id="edit-43"></a>
## 43. Fig. 1 图注

精简与主图内部结构重复的流程说明。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\captionof{figure}{Testing world models in the loop. A fixed planner flies three quadrotors carrying a slung load in a true simulator; the models under test plug into its world-model slot, and nothing else changes. Layers~1--3, illustrated in Fig.~\ref{fig:probes}, probe the model inside the planner against the true simulator; layer~4 measures closed-loop mission success. The pre-registered test asks whether the cross-agent layer explains success beyond open-loop prediction error.}
```

**修改后**

```latex
\captionof{figure}{World-model evaluation under a fixed planner and cost. Three quadrotors transport a slung load; only the predictor changes. Layers~1--3 use exact simulator references (Fig.~\ref{fig:probes}); layer~4 measures closed-loop mission success. The pre-registered test asks whether cross-agent error adds information beyond trajectory error.}
```

</details>

<a id="edit-44"></a>
## 44. Fig. 5 图注

不再逐项复述现有图例，保留统计对象和区间解释。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\caption{Mission success against layers 1--3, one point per model. Open-loop error (left) tracks success most closely. PETS (green) has a larger open-loop error than JEPA \texttt{flat} (blue downward triangles) but a smaller cross-agent error (middle), and flies more often. Blue marks JEPA (circles \texttt{full}, upward triangles \texttt{no\_cross}) and orange W-ar; filled markers are for 2,000 training episodes, open for 500. Gray squares mark the physics rows and the gray diamond the example MLP; the dashed line is the oracle U. Each panel gives the Spearman correlation $\rho$ across the 30 learned models with its 95\% interval, and the middle and right panels also the partial correlation given layer~1,~\eqref{eq:partial}.}
```

**修改后**

```latex
\caption{Mission success against layers~1--3, one point per model instance. PETS has larger open-loop error than JEPA \texttt{flat}, but smaller cross-agent error and higher success. Correlations use the 30 learned instances, excluding reference rows and the example MLP. Panels report Spearman correlations and 95\% intervals; the middle and right also report partial correlations given $L_1$,~\eqref{eq:partial}. Marker encodings are defined in the legend.}
```

</details>

<a id="edit-45"></a>
## 45. Fig. 6 图注

合并数据预算与分母说明，保留终止类别的边界。

<details>
<summary>展开：原文 → 修改后</summary>

**原文**

```latex
\caption{How the test episodes end, per row. Rows with three seeds pool their 1,200 episodes; the training-episodes column gives each world model's data budget. Nearly all of U's failures are timeouts; most of the learned models' and the policies' failures are collisions. Floor or workspace: an aircraft or the load hits the floor, or the team leaves the workspace.}
```

**修改后**

```latex
\caption{Test outcomes by method and training budget. Oracle U's failures are nearly all timeouts; learned models and policies fail mainly by collisions. Floor or workspace denotes aircraft or load contact with the floor, or workspace exit. Three-seed rows pool 1,200 episodes; other rows contain 400.}
```

</details>