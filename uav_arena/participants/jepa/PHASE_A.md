# JEPA Phase A：explore-v2

已实现 full / no_cross / flat 三个变体并完成五次训练与验证。当前最佳分数均高于8.26 cm，尚未运行dev。

冻结基线：arena-v0.3 (`64134e72f66be128024b1e602620073de4666851`)。
训练只用前250个episodes（200000–200249）；验证100个episodes（299000–299099），
四种latency、15步、官方openloop --stride 10。门槛严格为8.26 cm以下。
训练seed为0，当前阶段为Phase A。

| 实验 | history | steps | head_weight | SIGReg权重 | 验证cm | 门槛 | dev成功 |
|---|---:|---:|---:|---:|---:|---|---|
| phase_a_full_a | 4 | 5,000 | 5 | 0.09 | 9.8473 | 未通过 | 未运行（未过门槛） |
| phase_a_full_b | 1 | 10,000 | 5 | 0.09 | 9.3593 | 未通过 | 未运行（未过门槛） |
| phase_a_full_c | 1 | 10,000 | 20 | 0.01 | 8.9844 | 未通过 | 未运行（未过门槛） |
| phase_a_no_cross_c | 1 | 10,000 | 20 | 0.01 | 8.8980 | 未通过 | 未运行（未过门槛） |
| phase_a_flat_c | 1 | 10,000 | 20 | 0.01 | 9.2660 | 未通过 | 未运行（未过门槛） |

A/B是full上的探索，均保留。B同时改history和训练步数，不能据此归因于history。
C基于重建误差调整现有损失权重；三个变体使用同一C配置。loss公式、指标和门槛未改。
默认encoder2层、predictor4层、hidden128、每实体latent64、heads4、dropout0。
flat每帧latent256，匹配总latent预算。所有实验batch128、lr0.001、seed0。

官方分延迟payload均值（cm），每个延迟均评分785个决策点：

| 实验 | 0ms | 50ms | 100ms | 200ms |
|---|---:|---:|---:|---:|
| phase_a_full_a | 9.5542 | 9.7064 | 9.9052 | 10.2232 |
| phase_a_full_b | 9.0923 | 9.2130 | 9.3947 | 9.7372 |
| phase_a_full_c | 8.6701 | 8.8505 | 9.0190 | 9.3981 |
| phase_a_no_cross_c | 8.5837 | 8.7786 | 8.9520 | 9.2777 |
| phase_a_flat_c | 9.0419 | 9.1705 | 9.2810 | 9.5707 |

| 变体 | 当前验证最优checkpoint | 含readout参数 | 峰值显存MiB | 训练分钟 |
|---|---|---:|---:|---:|
| full | phase_a_full_c | 1,288,841 | 3642.1 | 34.71 |
| no_cross | phase_a_no_cross_c | 1,288,841 | 3642.1 | 33.01 |
| flat | phase_a_flat_c | 1,371,663 | 944.5 | 20.45 |

记录的Phase A训练GPU时间合计2.341小时；不含统计、验证、dev和调试。
设备RTX4060 Laptop 8GB，WSL2 Ubuntu26.04.1；时间受笔记本功耗和并发CPU检查影响。
配置、分延迟结果、checkpoint及source/data SHA256汇总于[phase_a_summary.json](results/phase_a_summary.json)。
训练生成的checkpoint与原始结果保存在`data/uav_arena/`下。

512候选×15步调用耗时：CUDA，候选从CPU输入并将结果还原到CPU，观察编码已缓存；
warmup2次后测量10次均值，包含设备传输及同步，不包含observe或完整MPPI循环。

| 变体 | 单次预测ms |
|---|---:|
| full | 25.84 |
| no_cross | 24.84 |
| flat | 23.90 |

## 误差诊断（不是官方性能分数）

固定前10个验证episodes的256个窗口，拆分读出重建与latent rollout误差。
“真实未来编码后重建”列只测encoder/readout；实际预测列从观测latent自回归生成。

| 变体 | 已知观测重建cm | 真实未来编码后重建cm | 实际latent rollout cm |
|---|---:|---:|---:|
| full | 3.70 | 5.34 | 9.66 |
| no_cross | 3.70 | 4.62 | 9.43 |
| flat | 5.15 | 6.68 | 9.87 |

读出重建本身仍有误差，latent rollout又增加误差；尚未定位到单一根因。
下一轮先分别检查读出重建和多步预测的收敛，再决定history与训练日程。

```bash
# 在仓库根目录；每次使用新的输出名。C配置三个变体只有attn不同。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.train \
  --episodes 250 --steps 10000 --batch 128 --history 1 --rollout-steps 15 \
  --attn full --head-weight 20 --sigreg-weight 0.01 --lr 0.001 --workers 4 \
  --seed 0 --device cuda --out data/uav_arena/participants/jepa/reproduce_full_c.pt
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/reproduce_full_c.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --stride 10 \
  --out data/uav_arena/results/reproduce_full_c_openloop.json
# 仅在selection_score < 0.0826后执行一次dev；不得用dev选配置。
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.run \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/reproduce_full_c.pt","device":"cuda"}' \
  --suite dev --seeds 0:5 --out data/uav_arena/results/reproduce_full_c_dev.json
```

## 实现与共享接口

- 基于`vmas_pusht/model.py`，使用latent MSE、SIGReg、encoded/predicted readout及delta loss。
- 输入状态51维；读出规划所需15维。encoder、predictor与readout联合训练，预测确定性。
- W-ar通过`uav_arena.participants.jepa.backbone:make_backbone`导入骨干，接口见[README.md](README.md)。
- 骨干默认history=4，当前C配置history=1。配对实验需同步history、骨干宽度并报告包含各自读出头的总参数量。
- no_cross阻断无人机之间的直接attention，保留经过payload的间接路径。

## 验证

相关102项测试通过，覆盖Arena回归、状态排列、15步反传、1/512/555候选、reset、
不读取载荷质量、时间因果性、坐标平移和checkpoint重载。完整测试命令与结果见结果JSON。

## 后续任务与验收

按[TASKS.md](../../TASKS.md)推进三条JEPA模型行，并与W-ar共享骨干。

### Phase A：训练收敛与闭环检查

- [x] 实现三个变体、共享骨干、训练入口、预测接口及测试。
- [x] 完成前250个训练episodes的五次试验，记录完整验证集的stride-10分数、参数量与计算开销。
- [ ] 分别检查readout重建和latent多步预测的逐步误差，再调整history、损失权重与训练日程。
- [ ] 将三个变体的验证selection score分别降到8.26 cm以下。
- [ ] 各变体运行一次dev seeds 0–4的全部8种条件（40 episodes），记录分条件成功数、失败类型与到达时间。
- [ ] 与june同步W-ar配对实验的history、full mask及骨干宽度，并报告包含各自读出头的总参数量。

### Phase B：维护者审阅后执行

- [ ] 三个变体使用同一组最多8个配置，在2,000个标准训练episodes上分别训练；
  用完整验证集的每个decision、四种延迟分别选出各变体的配置，并记录所有尝试。
- [ ] 每个变体固定所选配置，分别在2,000和前500个episodes上训练各3个新seed；
  两种数据预算保持相同训练步数，500-episode组沿用所选配置。三变体合计18个正式模型。
- [ ] 每个正式模型在冻结MPPI下执行一次test：50个layouts×8种条件=400 episodes，记录成功率J。
- [ ] 对接维护者的U记录状态与harness，补齐L1、L2跨智能体CEE与L3高代价决策比例，
  按[ANALYSIS.md](../../ANALYSIS.md)交付各条件、seed和数据预算的指标与layout bootstrap区间。
- [ ] 汇总参数量、训练GPU小时和512×15预测耗时；记录arena tag、harness commit、
  question-set manifest与checkpoint SHA256。

当前表格中的selection score用于训练配置选择；正式L1在U的记录状态上预测U的15步计划。
后续分析使用L1、L2和J检验跨智能体误差与闭环成功率的关系，并比较三种实体结构。
