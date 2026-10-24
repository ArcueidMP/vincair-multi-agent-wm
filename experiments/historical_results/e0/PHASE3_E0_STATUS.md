# E0 实验阶段状态

更新日期：2026-10-23。VMAS PushT 的 E0-A 至 E0-D 四组各 8 集诊断已完成；MultiPushT E0-D 已完成 150 集（95 个 must-move）。MultiPushT E0-A 当前累计 15 个 must-move 场景，成功 0/15；对应 E0-D 也是 0/15。E0-A 为非均衡 pilot，完整 95 集及 E0-B/C 尚未完成。

## MultiPushT E0-D：joint LeWM + native latent cost

使用作者 joint checkpoint、centralized CEM、budget 150、horizon 5、每 25 步重规划、300 samples × 30 iterations、30 elites、batch size 1。每个 seed 跑 eval 集 50 个场景；0–2 种子中 95 个场景需要移动 T，其他 55 个起始时已在目标容差内。下表中的必须移动成功数以 95 个场景为分母。

| Seed | 全部场景成功 | 必须移动 T 成功 | 初始已放置成功 | 墙钟时间 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 16/50 (32%) | 1/30 | 15/20 | 459.6 s |
| 1 | 21/50 (42%) | 1/28 | 20/22 | 419.1 s |
| 2 | 11/50 (22%) | 1/37 | 10/13 | 513.8 s |
| 合计 | 48/150 (32%) | 3/95 (3.16%) | 45/55 | 1392.6 s |

该结果显示模型在“已经放好”的初态上占多数成功；必须实际推动 T 的成功率明显低。它是当前 bounded-action 本地代码版本的结果，不能直接当作作者原始协议的严格复现。仓库 README 报告的 seed 0 joint checkpoint 历史值为 must-move 1/30、总计 2/95；当前 seed 0 的 must-move 也为 1/30，但其余口径与代码补丁条件不同，不据此声称相同或改进。

清单来自仓库 `assets/eval_episodes.npz`，SHA256 `0ba9207a0dd4c880a7fa0ab83eb49e4ba0fcb3f250075b334f0a2cb6b076aa19`。95 场景清单 SHA256 `25e7f6fe2a6ffc95f841e466dcf3330bfb66f00f5facddbd84ceec12482c70c3`。checkpoint SHA256 `7b975572b7542d65223139ffe21d30bfc6c454630d14e831dc24dfaa372f2a1c`。每个 seed 的 metadata 保存完整 argv、代码 diff 和资源信息；150 份逐步状态/动作轨迹在 `results/e0/multipusht_joint_seed{0,1,2}/`。

## MultiPushT E0-A 精确真仿真试跑

为保证候选动作序列从完全相同的 Pymunk 状态开始，每次候选都使用全新 reset 并重放已执行动作前缀；体状态断言误差为 0。Seed 0 第一个必须移动场景、budget 150 的单 episode 运行完成：0/1 成功，终态 T 位置误差 55.16 px，角度误差 0.00278 rad，总墙钟 1721.9 s（约 28.7 分钟），进程 RSS 峰值约 1357.6 MiB。单 episode 成本若线性外推，95 集约 45.4 CPU 小时；这只是粗略估算，尚未跑完整 E0-A。

本地对 body-only 快照方案的 10 步接触回放审计发现全身状态最大差异 0.02380、位置差异 0.01422 px，因此 E0-A 使用 reset+重放路线，不用 body-only 状态快照冒充精确仿真基线。

## VMAS E0-A 至 E0-D

固定 Seed 42 的同一 8 个场景，场景清单 SHA256 `0fd4be87a1519376f6c30828cf6050221d0d5d5104c7dd2ceadd48285ad0cf5b`。控制配置为 RoleSchedule（step 0.15）、joint CEM、lookahead 15、window 30、horizon 3（15 环境步）、每 15 环境步重规划、budget 600、300×30、30 elites、colored noise beta 2。模型组使用 `wm_full_h_agg2.pt`。

| 组别 | 动力学 | Cost | 成功 | 终态平均误差 | 双方同时接触 | 墙钟时间 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| E0-A | 真 VMAS | 真状态距离 | 0/8 | 0.5520 | 57.23% | 668.8 s |
| E0-B | 真 VMAS | predicted-state probe / decoded cost | 1/8 | 0.5311 | 47.75% | 732.9 s |
| E0-C | learned WM | 同一 decoded Readout Cost | 3/8 | 0.4589 | 22.94% | 92.7 s |
| E0-D | learned WM | native latent cost | 0/8 | 0.6874 | 6.42% | 86.9 s |

Readout 仅使用固定 HF revision `d5e1a7ac841ad859e91de6bcc48fe5abb763bba4` 的 `train.pt` 拟合；`heldout.pt` 未用于拟合或调参。E0-C 曾按用户要求中断后重跑；首次中断未产生结果或轨迹，metadata 已被成功重跑覆盖，不纳入分析。

## 可解释范围与当前结论

- 所有结果都是诊断性小样本；不得据此宣称统计显著或方法优劣。
- MultiPushT 3/95 的 must-move 成功率低，且 E0-D 使用 bounded-action 本地修正；需先诊断规划搜索、碰撞接触和动作序列，再决定 E1 或 E2/E3/E5/E6 的优先级。
- MultiPushT E0-A 不是 95 集完整基线；E0-B/C 的对应 probe/readout 尚未训练或评估，不能把 VMAS 的 A–D 结果替代 MultiPushT 对照。
- 尚未从当前可用结果证明低成功率由搜索、动力学模型或 cost 单独导致。不要对同一 95 场景重复调参后再把它作为独立测试集。
- MultiPushT 相关模型权重未在此处重新下载；报告只引用已有本地 checkpoint 的 hash，不表示仓库/权重当前已完整备份到本机其他位置。

## 下一步

按 `Multi_Agent_LeWM.md` 的顺序先做 E1 候选/动作搜索诊断：固定 95 个必须移动场景清单，明确评估预算与硬件时间，再运行少量 scene 的候选质量、动作合法性、接触覆盖和重规划消融；将探索集与最终评估集分开。之后依据诊断选择 E2/E3/E5/E6。开始完整 E0-A 或训练 MultiPushT Probe 前，先估算耗时和训练数据划分；当前没有启动训练，也没有声称 MultiPushT 的 E0 完成。

## 最新执行记录：MultiPushT Readout 试跑与 E0-A 暂停

按计划尝试了 MultiPushT readout 准备：使用新生成的 10 个训练场景（seed 20261022；不含 95 个 eval 场景），20 条 agent-view episodes、3,994 帧，训练/验证按 scene 划分为 8/2。试拟合 encoder latent 与一步 predicted latent 两种 readout，预测 latent 的 held-out 验证位置 RMSE 为 160.73 px、角度 MAE 为 69.44°；训练场景误差低但验证误差很高，说明 10 场景试采集明显不足。该 readout 只作管线诊断，不用于 E0-B/C。数据、试拟合权重和 metadata 位于 `data/datasets/multipusht_readout_pilot_seed20261022.h5` 与 `results/e0/multipusht_readout_pilot/`。

随后启动不依赖 readout 的 E0-A 真仿真 smoke 批次，固定 Seed 0 must-move episode 0、1、4、5、6。episode 0 是已完成精确 pilot（0/1，终态 T 位置误差 55.16 px，耗时 1721.8 s）；episode 1 在用户要求暂停时已执行 125/150 环境步，处于最后重规划窗口第 15/30 次 CEM 迭代。按请求停止，未保存该 episode 的结果或轨迹，metadata 标记为 `interrupted_by_user`；后续 episode 4、5、6 未启动。episode 1 不计入 E0-A 成功率。完整 E0-A、E0-B/C 仍未完成。

## E0-A smoke 批次完成（2026-10-23）

恢复后从头完整重跑 seed 0 episode 1，并继续完成 episode 4、5、6；连同此前完成的 episode 0 pilot，共评估 5 个预先固定的 must-move 场景（IDs 0、1、4、5、6）。结果 0/5 成功。各集终态 T 位置误差分别为 55.16、86.84、7.12、80.58、30.23 px；角度误差为 0.0028、0.9959、1.4862、0.7350、0.2114 rad。双 agent 同步接触比例仅 episode 5 为 6%，其余为 0。五集逐候选精确分支状态误差均为 0，轨迹状态/动作有限，动作保持在 [-1,1]。

五集墙钟时间分别为 1721.84、1602.37、1520.67、1552.23、1728.62 秒，合计约 2.26 小时，均值约 27.1 分钟/集。该 smoke 样本仍不能代表完整 95 集。旧 episode 1 中断尝试的 metadata 保存在 `metadata_interrupted_20261022.json`，该次 125 步未作为结果；重跑数据才是本批正式 episode 1。Seed 0 的 E0-A 当前有 5 个有效场景记录，尚缺该 seed 的其余 must-move 场景及 seed 1、2 的对应场景。完整结果位于 `results/e0/multipusht_true_true_pilot/seed0_ep{00,01,04,05,06}/`。

## E0-A 扩展到十集（更新：2026-10-23）

seed 0 的固定 must-move 场景 IDs `0,1,4,5,6,8,10,12,13,14` 全部完整评估，成功 **0/10**。平均终态 T 位置误差 **56.32 px**、平均角度绝对误差 **0.565 rad**；双方同一步接触比例按 episode 平均 **0.8%**。单集墙钟均值 **1592.1 s（26.5 分钟）**，合计 **4.42 小时**。十集动作/状态均有限且边界通过，所有精确分支重建最大误差为 0。低同步接触比例使协调/接触不足成为值得检查的失败线索；0/10 仍不能单独证明其为唯一原因。

这十集只覆盖 seed 0 的 30 个 must-move 场景中的前 10 个；seed 0 剩余 20 个及 seed 1/2 尚未完成。若按当前单集成本线性外推，剩余 85 集约需 37.6 小时，实际耗时会变化。结果、metadata 与轨迹在 `results/e0/multipusht_true_true_pilot/seed0_ep{00,01,04,05,06,08,10,12,13,14}/`；episode 1 的上次中断尝试单独保存在 `seed0_ep01/metadata_interrupted_20261022.json`，不计入结果。

## E0-A 批次暂停与替代方案（2026-10-23）

因完整 95 集成本过高，已停止 seed 0 episode 18；该集未完成环境步，也没有结果或轨迹，不纳入统计。episode 17 已完成并保留（0 成功，耗时 1576.9 s）。本批剩余 episode 未启动。当前累计 11 集、0/11 成功；全量 E0-A 尚未完成，剩余 84 集按当前均值粗估约 37.1 小时。

### 可选执行路径

1. **推荐：转为 E1 前的失败机制诊断。** 保留现有 11 集作为基线；围绕低同步接触（0.8%）检查规划候选是否能接近双侧接触、分阶段接触和有效推动，并用少量独立场景诊断。完成后进入计划中的 E1 搜索/协调分析。这样把资源用在解释 0/11 失败，而不是再重复相似的昂贵失败集。E0-A 明确标为 seed0 的 11 集 pilot，不能写成 95 集基线。
2. **如需跨 seed 稳健性：** seed 1 与 seed 2 各选 2–3 个固定 must-move 场景，共新增 4–6 集，按实测约 1.8–2.7 小时；使用原始完整 CEM 预算。它只检验失败现象是否跨 seed 出现，仍不替代 95 集评估。
3. **如必须覆盖全部 95 场景：** 建立单独的低预算筛查协议，例如 CEM 100 samples × 10 iterations（当前是 300 × 30），先在 1 个未用场景上计时和验证轨迹，再决定是否扩展。候选评估数理论上约为原协议的 1/9；若运行时近似随候选数变化，95 集约需 4–5 小时，但这是未验证估算。它属于新协议；要与 E0-D 做公平 A/D 对照，必须在同场景、同低预算下补跑 E0-D，不能直接拿现有 E0-D 数值比较。

建议采用路径 1；若需要给失败现象补跨 seed 证据，再增加路径 2 的 4–6 集。无论选哪条，都保留 E0-A 为未完成的 pilot，并在报告中把协议变化、分母和耗时写清楚。没有启动新的实验。




## E0-A 累计 15 集完成（2026-10-23）

用户授权补充的 seed1/2 episode0/1 已全部完成；四进程并发实际墙钟 30.78 分钟。E0-A 累计 0/15，相同 15 个场景的 E0-D 为 0/15；按 seed 为 0/11、0/2、0/2。A 平均位置误差 58.30 px、角度误差 0.580 rad、同步接触比例 0.533%。15 集均通过状态/动作有限值、动作边界、精确分支重建（最大误差 0）和 A/D 初态逐元素一致性核验。seed0_ep18 中断记录不计入。

详见 results/e0/multipusht_e0a_15/E0A_15_REPORT.md；配对明细和 summary、选样清单、日志已保存。此为非均衡 15 集 pilot，完整 95 集 E0-A 及 E0-B/C 尚未完成；A/D 随机流和 cost 不同，不作单因素归因或优劣结论。当前批次已完成并停止，不自动进入下一实验；建议下一步 E1 搜索/接触诊断。


## 后续阶段（2026-10-23）

E1 数据准备与排序诊断 pilot 已完成，8 个新场景、47 状态、752 候选，未训练 Ranker。见 ../e1/ranking_pilot_v1/E1_PILOT_REPORT.md。E0状态不变：MultiPushT E0-A 0/15 pilot，完整95集及E0-B/C尚未完成。

## 正式 readout 最新进度（2026-10-23）

Readout已到位并完成752候选真实/预测latent核验。另完成两场景低预算A/B/C，各0/2，协议150步/H10/16×3，独立核验通过。原完整预算E0-A保持0/15，两套协议不合并；完整E0-B/C尚未完成。详见 readout_smoke_v1/READOUT_SMOKE_REPORT.md。作者拟合数据已下载并完成初态重合筛查，见 ../readout_intake_20261023/scene_overlap.json。
