# MultiPushT 正式 Readout：独立误差核验与低预算 A/B/C 试跑

日期：2026-10-23。正式 joint probe 已接入诊断流程。已完成离线核验及两个固定 must-move 场景的 A/B/C 管线试跑，未进行训练。该低预算试跑不等于原方案完整 E0-B/C。

## 1. 同候选离线比较

复用 E1 pilot 的 8 个独立场景、47 个状态、752 条 10 步候选。每条候选重新 reset 并重放原始历史动作，真实终态与此前保存结果逐元素一致。分别对真实终态图像编码和模型预测 latent 使用同一个正式 joint probe，目标均从同一目标图像编码读出。

Cost 沿用本地 E0-A 的联合位置距离范数 + 加权环形角度绝对误差，不采用上游 squared pose_cost。模型与 probe 均冻结，权重和脚本 hash 已记录。

### Pilot validation（2 场景、11 状态、176 候选）

| 指标 | 真实终态图像 latent 的 readout | 预测 latent 的 readout |
| --- | ---: | ---: |
| Self 位置中位误差 | 10.78 px | 16.06 px |
| T 位置中位误差 | 10.02 px | 10.55 px |
| T 位置 RMSE | 12.41 px | 14.43 px |
| T 角度中位误差 | 1.42° | 1.83° |
| 成对排序准确率（PRA） | 87.31% | 79.63% |
| 状态平均 Spearman | 0.785 | 0.676 |
| Top-1 真实 cost regret | 0.232 | 0.971 |

同一批验证候选上，之前原生预测 latent cost 的 PRA 为 56.09%。正式 readout 的离线排序数值更好；但仅来自两个验证场景，不能据此宣布统计显著或闭环成功率改善。

误差指标将候选终态汇总计算；排序指标先按状态计算再等权平均。每个状态内的候选及同场景内状态相互相关，176 条候选不是 176 个独立场景。PRA 忽略真实 cost 差不超过 1.0 的候选对。

Pilot train（6 场景、36 状态）的真实/预测 latent readout PRA 分别为 91.03% / 81.63%；这些场景在本阶段也未用于拟合或调参。详细位置、角度和 regret 指标见 audit_summary.json。

解读：短期预测 latent 上确有额外读出误差，但当前 10 步下没有出现此前本地失败 probe 的百像素级误差。不能外推到 25 步或更长 rollout。B/C 的 goal 同时采用读出值，因此仍需保留目标误差；它不是“真实 goal 完全正确”的对照。

## 2. 两场景低预算闭环试跑

在查看本次结果前，固定选择 seed1 episode0 和 seed2 episode0，均为已审计的 must-move 场景。这是调试样本，不再作为未见的最终测试集。A/B/C 同初态、同每集 RNG seed、同 success test、同 cost 公式：

- A：真实候选 rollout + 真实 endpoint pose，真实 goal。
- B：真实候选 rollout 后渲染/编码 + 正式 probe，图像读出 goal。
- C：学习模型 rollout + 同一正式 probe，同一图像读出 goal。

三组统一 150 环境步、10 步 horizon/重规划周期、16 samples × 3 iterations、4 elites，std floor 0.05，动作边界 [-1,1]。与此前 300×30、25 步 horizon 的 15 集 E0-A 协议不同，不能直接合并统计。A/B 评分不同后会产生不同轨迹，后续优化均值与候选也随之不同；并非整集始终使用同一候选集合。

| 场景 | 组别 | 成功 | T 位置误差 px | T 角度误差 rad | 真实终态 cost | 单集墙钟 s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| seed1 ep0 | A | 否 | 78.92 | 0.7065 | 119.47 | 17.91 |
| seed1 ep0 | B | 否 | 78.92 | 0.7065 | 121.83 | 22.50 |
| seed1 ep0 | C | 否 | 78.69 | 0.7032 | 129.13 | 1.56 |
| seed2 ep0 | A | 否 | 57.25 | 0.8865 | 152.51 | 21.12 |
| seed2 ep0 | B | 否 | 59.40 | 0.9116 | 120.77 | 24.42 |
| seed2 ep0 | C | 否 | 58.02 | 0.8909 | 115.85 | 1.42 |

三组均为 0/2，所有六集的同控制步双方接触比例均为 0。Cost 包含 agent0 的目标误差，所以不能只用 T 位置误差推断 cost 排名。真实接触标记由一个控制步内的碰撞回调累计，不能等同于物理子步同时有效施力。

本阶段离线核验加六集试跑的脚本内部计时 116.65 秒，包含模型加载，不含模块导入、编写脚本和独立检查。PyTorch 张量分配峰值约 126.4 MiB，不代表整卡显存占用。

这批结果说明 B/C 的接口和闭环执行管线已跑通；较好的离线评分尚未在这两个低预算场景转化为成功。A 也失败、样本量和搜索预算均小，不能判断正式 readout 无效，或唯一瓶颈已被识别。原完整预算 E0-A 仍保持 0/15，本批 A 不加入该统计。

## 3. 独立核验

validation.json 为 passed：

- 752 个候选的真实/读出 cost 重新计算一致，数组有限。
- 6 集初态、合法动作、真实状态、接触记录及成功判据均独立核验。
- A/B/C 对应初态逐元素相等，B/C goal pose 逐元素相等。
- 每集真实执行动作从原始 reset 重放，全部状态及接触与记录一致，最大状态误差为 0。
- A/B 候选分支完整刚体重建最大误差为 0；C 使用学习模型评分，不将 C 结果内的初始化为 0 的分支字段当作动力学准确性证明。
- Probe、基础模型、运行脚本 hash 一致，未覆盖任何既有 checkpoint。

## 4. 作者拟合数据与重合核验

已从同一固定 HF revision 下载 `datasets/multipusht_2a_coop_heldout.h5`，699,370,403 字节，SHA256 与 HF LFS 一致。实际包含 100 个场景，按作者 seed0 split 规则得到 70 train / 10 val / 20 test。

比较 agent0 xy、T xy、T angle 的初始状态（容差 0.001，忽略伙伴位置），未发现与本地 150 个 E0 场景、或本地 8 个独立 pilot 场景重合。这是初态身份筛查，不是全轨迹去重，也不能证明作者原 WM 预训练数据无重合。未重新训练 probe，亦未完整复现作者 test 指标。

核验记录：`results/readout_intake_20261023/scene_overlap.json`。该公开数据已取得，无需再向组员索取。

## 5. 当前进度与建议

正式 MultiPushT readout 的材料阻塞已解除。E0-B/C 完成了两场景低预算管线试跑，完整协议评价仍未完成；E1-B 现在有可用的离线 readout 对照，E1 Ranker 尚未训练。

下一步优先检查接触到位与局部目标/Worker 协议，确认是否应复用团队已有 Coordinator。若做 Ranking Head，必须把正式 readout cost 加为强基线，不能只与 56.1% 的 latent cost 比较。继续保留独立场景划分，闭环调试样本不再充当最终未见测试集。

如需复现组会中已报告的 MultiPushT 分层规划成果，需组员确认并提供对应 Coordinator/RoleSchedule、Worker 代码 commit、运行配置和日志。当前已验证的本地 MultiPushT 规划为 flat CEM；公开 CoopPolicy 是读取真状态的数据采集参考，不能代替所需的分层规划协议。

后续缺失权重、配置、数据或协议时，按用户要求立即列明具体缺项、版本、用途和受影响实验；能从已确认的公开来源取得的材料由本地直接获取。

## 6. 产物

主目录：`D:/资料库/03_开发项目/air traffic/experiments/lewm/results/e0/readout_smoke_v1/`。

- metadata.json：固定场景、预算、cost/goal 来源、权重和代码 hash。
- audit_scene*.npz：逐状态真实/预测 readout、物理标签和 goal。
- audit_state_metrics.json、audit_summary.json：离线结果。
- A/B/C_seed*/：6 份逐集 metadata、results、trajectory、规划窗口 cost。
- pilot_results.json、validation.json：闭环汇总与独立验证。

脚本：`scripts/phase5_readout_smoke.py`、`scripts/validate_readout_smoke.py`、`scripts/check_probe_scene_overlap.py`。
