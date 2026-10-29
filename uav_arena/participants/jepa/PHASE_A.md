# JEPA Phase A：explore-v2

已实现 full / no_cross / flat 三个变体并完成六次训练与验证。当前最佳分数均高于8.26 cm，尚未运行dev。

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
| phase_a_full_d_fresh_v1 | 1 | 23,000 | 20 | 0.01 | 8.3737 | 未通过 | 未运行（未过门槛） |

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
| phase_a_full_d_fresh_v1 | 8.0500 | 8.2423 | 8.4152 | 8.7871 |

| 变体 | 当前验证最优checkpoint | 含readout参数 | 峰值显存MiB | 训练分钟 |
|---|---|---:|---:|---:|
| full | phase_a_full_d_fresh_v1 | 1,288,841 | 3642.1 | 75.72 |
| no_cross | phase_a_no_cross_c | 1,288,841 | 3642.1 | 33.01 |
| flat | phase_a_flat_c | 1,371,663 | 944.5 | 20.45 |

六次完成的Phase A联合训练GPU时间合计3.603小时；不含统计、验证、探针及调试。
另保留原D训练中断（磁盘权重10,000步、日志10,400步）和warm-start中断（新增4,185步）的记录，不计入完成模型时间或分数。
设备RTX4060 Laptop 8GB，WSL2 Ubuntu26.04.1；时间受笔记本功耗和并发CPU检查影响。
六次完成实验的配置与完整指标保存在`data/uav_arena/results/jepa/phase_a_summary.json`；checkpoint SHA256见下表。
训练生成的checkpoint与原始结果保存在`data/uav_arena/`下。

| checkpoint | SHA256 |
|---|---|
| phase_a_full_a | `0cf9f51fcfc6e656de188ff3e93335fe8680c96cb4d243cea8556a752a91ce0b` |
| phase_a_full_b | `774f30d551d2eaaa04cf45febe9328807758e2da782949964ff9663cc10ff7b7` |
| phase_a_full_c | `6461507f53be90b700e73428fb6652a3a96573b3b8de37a2299b5ff51084fbdb` |
| phase_a_no_cross_c | `1847a74039bdc0ad0717bddd4bd69aa14c1c7643754f8c92410c45f89c31233d` |
| phase_a_flat_c | `b7842f3697f0e1780fc22df6a610f1b2dfb1d011c762a468b39dcaf9102dba0a` |
| phase_a_full_d_fresh_v1 | `ba08fccd69c7fe0b2ca4c65a00d0721e5d651b7d0a89bc47b8124840832d4b43` |

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


## 数据校验

在当前arena-v0.3数据上重新执行`python -m uav_arena.collect --verify <directory>`：

| 数据集 | match | differ | unknown |
|---|---:|---:|---:|
| explore_v2 | 2000 | 0 | 0 |
| explore_v2_val | 100 | 0 | 0 |

训练仍只选取前250个episodes。

## 实现与共享接口

- 基于`vmas_pusht/model.py`，使用latent MSE、SIGReg、encoded/predicted readout及delta loss。
- 输入状态51维；读出规划所需15维。encoder、predictor与readout联合训练，预测确定性。
- W-ar通过`uav_arena.participants.jepa.backbone:make_backbone`导入骨干，接口见[README.md](README.md)。
- 骨干默认history=4，当前C配置history=1。配对实验需同步history、骨干宽度并报告包含各自读出头的总参数量。
- no_cross阻断无人机之间的直接attention，保留经过payload的间接路径。

## 冻结编码器读出探针

父checkpoint：`phase_a_full_c.pt`。冻结完整父模型，另行随机初始化同结构读出头，
使用前250个训练episodes的全部67,012个窗口训练5,000步，batch128、lr0.001、
AdamW（weight decay0.001）、OneCycle、seed0。沿用父模型归一化，只优化编码状态读出的MSE。

完整100个验证episodes按stride10评估3,140个窗口，覆盖四种延迟。
编码器、预测器及原读出头参数的训练前后哈希一致，父checkpoint文件也保持不变。

| 划分 | 读出头 | 观测载荷cm | 未来载荷cm | 观测飞机cm | 未来飞机cm | 未来载荷速度m/s |
|---|---|---:|---:|---:|---:|---:|
| 训练集 | 原读出 | 3.296 | 4.800 | 8.871 | 10.027 | 0.0847 |
| 训练集 | 新读出 | 1.443 | 1.942 | 2.962 | 3.501 | 0.0420 |
| 验证集 | 原读出 | 3.187 | 4.883 | 8.579 | 9.975 | 0.0858 |
| 验证集 | 新读出 | 1.442 | 1.995 | 2.916 | 3.560 | 0.0432 |

验证集未来载荷重建误差分段：

| 读出头 | 延迟 | 第1–5步cm | 第6–10步cm | 第11–15步cm |
|---|---|---:|---:|---:|
| 原读出 | 0 ms | 4.497 | 4.751 | 5.188 |
| 原读出 | 50 ms | 4.587 | 4.778 | 5.216 |
| 原读出 | 100 ms | 4.637 | 4.816 | 5.247 |
| 原读出 | 200 ms | 4.648 | 4.907 | 5.327 |
| 新读出 | 0 ms | 1.567 | 1.970 | 2.288 |
| 新读出 | 50 ms | 1.606 | 1.988 | 2.318 |
| 新读出 | 100 ms | 1.650 | 2.002 | 2.346 |
| 新读出 | 200 ms | 1.741 | 2.050 | 2.416 |

这次完整验证集上的原读出观测载荷误差为3.19 cm；前述3.70 cm诊断使用10个episodes、256个窗口。
新读出恢复了相当部分位置精度，但未来载荷误差仍约2.00 cm，尚未达到1 cm。
训练与验证误差接近；本次5,000步同容量探针仍不能单独判定剩余误差来自编码精度还是读出优化收敛。
观测帧载荷相对位置恒为零，因此判断同时使用飞机位置和未来帧。

探针只测试真实帧编码后的重建，未替换参赛者读出头；原模型selection score仍为8.9844 cm。
后续full的23,000步联合训练已完成，新增诊断见下一节。逐步归一化与目标尺度修改仍保留为讨论项。

探针耗时51.92秒，完整指标与独立权重保存在：

- `data/uav_arena/results/jepa/full_c_readout_probe_v1.json`
- `data/uav_arena/results/jepa/full_c_readout_probe_v1.probe.pt`
- 探针权重SHA256：`c2330d220c709b3e84e12f7c871b84d4a0afd7eb621b87782da8c832fff8c3ba`
- 父模型冻结状态SHA256：`9d809bfae247d6ebb55add2d10d66746651b3a52d7976cbcb4ea3077b4bbc6d6`

复现命令见[README.md](README.md#frozen-encoder-readout-probe)。

## full D：23,000步后的诊断与结论

从随机初始化开始，沿用C的模型、损失、归一化统计和seed0；训练器新增完整断点，保持原shuffle顺序。完整23,000步OneCycle训练耗时75.72分钟。选择分数8.3737 cm，较full C改善0.6108 cm，仍高于门槛0.1137 cm。

四种延迟平均的载荷误差：

| 阶段 | 示例cm | full C cm | full D cm |
|---|---:|---:|---:|
| 第1–5步 | 4.6598 | 6.0980 | 4.9017 |
| 第6–10步 | 9.7406 | 9.8301 | 9.5616 |
| 第11–15步 | 10.3774 | 11.0253 | 10.6577 |

中段已优于示例，短期和末段仍偏高；飞机误差在四种延迟下改善但仍高于示例。

相同5,000步冻结编码器探针，在完整验证集上：

| 模型 | 原读出未来载荷cm | 新读出未来载荷cm | 原读出未来飞机cm | 新读出未来飞机cm |
|---|---:|---:|---:|---:|
| C | 4.8832 | 1.9952 | 9.9752 | 3.5596 |
| D | 2.2120 | 1.6358 | 4.9968 | 2.0780 |

训练量增加已显著改善读出精度，新读出头仍有收益，但尚未达到1 cm。探针仅重建真实编码，不替换官方预测接口，不能据此推算选择分数。新探针耗时39.68秒，冻结参数与父文件哈希验证通过。

参数量×训练采样次数：D为3.7943e+12，示例为3.8599e+12，比例0.983；该口径不是FLOPs。

逐步目标归一化已整理为待讨论方案，尚未修改损失。建议先确认是否只重加权predicted未来读出的absolute项，保持encoded读出、delta、latent、SIGReg及骨干不变。详见[完整诊断与讨论](DIAGNOSTICS_FULL_D.md)，含四种延迟分段、逐步误差、探针、实际训练统计尺度与复现口径。

原D中断和warm-start中断的权重、日志均保留；本表只将完整从头训练的D计为第六个完成实验。状态登记脚本的stride字符串类型检查已修正，未改变训练或评分结果。

从头复现D：使用上方C训练命令，把`--steps 10000`改为`--steps 23000`，并将所有输出前缀替换为新名称；其他参数保持一致。探针命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.probe_readout \
  --checkpoint data/uav_arena/participants/jepa/phase_a_full_d_fresh_v1.pt \
  --episodes 250 --val-episodes 100 --steps 5000 --batch 128 --lr 0.001 --seed 0 --device cuda \
  --out data/uav_arena/results/jepa/reproduce_full_d_readout_probe
```

完整原始结果继续留在`data/`：

- `data/uav_arena/results/phase_a_full_d_fresh_v1_openloop.json`
- `data/uav_arena/results/jepa/full_d_fresh_v1_readout_probe.json`
- `data/uav_arena/results/jepa/full_d_fresh_v1_diagnostics.json`
- 新探针SHA256：`b0a9e5ccfdb142c38790b28f8ba9d85ce690ebcc880f0af1011b760b1461a7c9`
- 父模型冻结状态SHA256：`b05bda2fb9f9294a1eb8201ea5869a263e6cc22088c486cdd96bd054ac954fe0`

## 验证

相关112项测试通过、1项跳过（在106项基础上新增2项续训和4项完整断点检查），覆盖Arena回归、状态排列、15步反传、1/512/555候选、reset、
不读取载荷质量、时间因果性、坐标平移和checkpoint重载。新增检查包括冻结参数、物理单位、逐episode验证stride、训练/验证划分、独立权重保存、原始shuffle顺序和跨epoch完整恢复。

## 后续任务与验收

按[TASKS.md](../../TASKS.md)推进三条JEPA模型行，并与W-ar共享骨干。

### Phase A：训练收敛与闭环检查

- [x] 实现三个变体、共享骨干、训练入口、预测接口及测试。
- [x] 完成前250个训练episodes的六次试验，记录完整验证集的stride-10分数、参数量与计算开销。
- [x] 完成full C/D的读出探针与官方多步误差对照，记录增加训练预算的结果。
- [ ] 与维护者确认逐步目标归一化的范围，再运行单一改动的受控实验。
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
