# JEPA Phase A：explore-v2

已实现 full / no_cross / flat 三个变体并完成七次训练与验证。当前full最佳分数高于精确门槛8.2593 cm；维护者2026-10-30批准带差距完成Phase A收尾，三组dev共120局已完成，Phase B尚未正式启动。

冻结基线：arena-v0.3 (`64134e72f66be128024b1e602620073de4666851`)。
训练只用前250个episodes（200000–200249）；验证100个episodes（299000–299099），
四种latency、15步、官方openloop --stride 10。精确门槛为selection_score < 0.082593 m（8.2593 cm）；8.26 cm仅为四舍五入显示。
训练seed为0，当前阶段为获例外许可的Phase A收尾。

| 实验 | history | steps | head_weight | SIGReg权重 | 验证cm | 门槛 | dev成功 |
|---|---:|---:|---:|---:|---:|---|---|
| phase_a_full_a | 4 | 5,000 | 5 | 0.09 | 9.8473 | 未通过 | 未运行（未过门槛） |
| phase_a_full_b | 1 | 10,000 | 5 | 0.09 | 9.3593 | 未通过 | 未运行（未过门槛） |
| phase_a_full_c | 1 | 10,000 | 20 | 0.01 | 8.9844 | 未通过 | 未运行（未过门槛） |
| phase_a_no_cross_c | 1 | 10,000 | 20 | 0.01 | 8.8980 | 历史未过线（现不单独要求） | 8/40 |
| phase_a_flat_c | 1 | 10,000 | 20 | 0.01 | 9.2660 | 历史未过线（现不单独要求） | 0/40 |
| phase_a_full_d_fresh_v1 | 1 | 23,000 | 20 | 0.01 | 8.3737 | 未通过；获例外收尾许可 | 28/40 |
| phase_a_full_e_40000_v1 | 1 | 40,000 | 20 | 0.01 | 8.4265 | 未通过 | 不用于dev |

A/B是full上的探索，均保留。B同时改history和训练步数，不能据此归因于history。
C基于重建误差调整现有损失权重；三个变体使用同一C配置。loss公式和评分指标保持原样；当前精确门槛按维护者后续澄清执行。
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
| phase_a_full_e_40000_v1 | 8.0847 | 8.3009 | 8.4947 | 8.8257 |

| 变体 | 当前验证最优checkpoint | 含readout参数 | 峰值显存MiB | 训练分钟 |
|---|---|---:|---:|---:|
| full | phase_a_full_d_fresh_v1 | 1,288,841 | 3642.1 | 75.72 |
| no_cross | phase_a_no_cross_c | 1,288,841 | 3642.1 | 33.01 |
| flat | phase_a_flat_c | 1,371,663 | 944.5 | 20.45 |

七次完成的Phase A联合训练GPU时间合计5.799小时；不含统计、验证、探针及调试。
另保留原D训练中断（磁盘权重10,000步、日志10,400步）和warm-start中断（新增4,185步）的记录，不计入完成模型时间或分数。
设备RTX4060 Laptop 8GB，WSL2 Ubuntu26.04.1；时间受笔记本功耗和并发CPU检查影响。
七次完成实验的配置与完整指标保存在`data/uav_arena/results/jepa/phase_a_summary.json`；checkpoint SHA256见下表。
训练生成的checkpoint与原始结果保存在`data/uav_arena/`下。

| checkpoint | SHA256 |
|---|---|
| phase_a_full_a | `0cf9f51fcfc6e656de188ff3e93335fe8680c96cb4d243cea8556a752a91ce0b` |
| phase_a_full_b | `774f30d551d2eaaa04cf45febe9328807758e2da782949964ff9663cc10ff7b7` |
| phase_a_full_c | `6461507f53be90b700e73428fb6652a3a96573b3b8de37a2299b5ff51084fbdb` |
| phase_a_no_cross_c | `1847a74039bdc0ad0717bddd4bd69aa14c1c7643754f8c92410c45f89c31233d` |
| phase_a_flat_c | `b7842f3697f0e1780fc22df6a610f1b2dfb1d011c762a468b39dcaf9102dba0a` |
| phase_a_full_d_fresh_v1 | `ba08fccd69c7fe0b2ca4c65a00d0721e5d651b7d0a89bc47b8124840832d4b43` |
| phase_a_full_e_40000_v1 | `950787015c95233805a5f97f97ec6997c9f96d30a4f8393cce7406666542a19d` |

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
此处为早期诊断；后续C/D探针与23,000步结果见下文。40,000步实验已完成，当前按2026-10-30评论完成收尾诊断。

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
# 2026-10-30例外收尾许可：dev指定full D与no_cross/flat C；不使用此处full C或full E。
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
后续full的23,000步联合训练已完成，新增诊断见下一节。目标尺度方案按维护者2026-10-29评论暂停，Phase A保留原配方。

探针耗时51.92秒，完整指标与独立权重保存在：

- `data/uav_arena/results/jepa/full_c_readout_probe_v1.json`
- `data/uav_arena/results/jepa/full_c_readout_probe_v1.probe.pt`
- 探针权重SHA256：`c2330d220c709b3e84e12f7c871b84d4a0afd7eb621b87782da8c832fff8c3ba`
- 父模型冻结状态SHA256：`9d809bfae247d6ebb55add2d10d66746651b3a52d7976cbcb4ea3077b4bbc6d6`

复现命令见[README.md](README.md#frozen-encoder-readout-probe)。

## full D：23,000步后的诊断与结论

从随机初始化开始，沿用C的模型、损失、归一化统计和seed0；训练器新增完整断点，保持原shuffle顺序。完整23,000步OneCycle训练耗时75.72分钟。选择分数8.3737 cm，较full C改善0.6108 cm，按澄清后的精确门槛8.2593 cm计算，仍高于门槛约0.1144 cm。

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

逐步目标归一化方案已由维护者明确暂缓；后续E按原配方、seed0完成40,000步，最新收尾许可见下文。详见[完整诊断与讨论](DIAGNOSTICS_FULL_D.md)，含四种延迟分段、逐步误差、探针、实际训练统计尺度与复现口径。

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

## full E：40,000步结果（未过线；后续收尾使用D）

按维护者要求，使用D的命令，仅将steps改为40,000、输出名改为`phase_a_full_e_40000_v1`。seed0，前250个训练episodes，batch128、history1、full、原损失和归一化均保持不变。最终模型参数、归一化统计和训练源码哈希已核对。

完整100个验证episodes、四种延迟、15步、官方stride10的selection score为 **8.4265 cm**。精确门槛为严格低于8.2593 cm（0.082593 m），本轮仍高0.1672 cm；相比23,000步D的8.3737 cm高0.0528 cm。当前full最佳仍为D。该轮未跑dev、不换seed、不追加训练；后续按2026-10-30例外许可用D/C模型完成收尾。

| 延迟 | 示例载荷cm | D载荷cm | E载荷cm | D飞机cm | E飞机cm |
|---|---:|---:|---:|---:|---:|
| 0ms | 7.8782 | 8.0500 | 8.0847 | 9.2672 | 8.6783 |
| 50ms | 8.0817 | 8.2423 | 8.3009 | 9.4036 | 8.8417 |
| 100ms | 8.3197 | 8.4152 | 8.4947 | 9.5113 | 8.9766 |
| 200ms | 8.7574 | 8.7871 | 8.8257 | 9.7399 | 9.2440 |

四种延迟平均的载荷误差分段：

| 预测阶段 | 示例cm | D cm | E cm | E−D cm |
|---|---:|---:|---:|---:|
| 第1–5步 | 4.6598 | 4.9017 | 4.7951 | -0.1066 |
| 第6–10步 | 9.7406 | 9.5616 | 9.6923 | +0.1307 |
| 第11–15步 | 10.3774 | 10.6577 | 10.7922 | +0.1344 |

E的前段改善，但中后段误差升高；不能仅凭训练目标下降断定具体原因。D/E最终训练目标分别为1.4544/0.8283，官方验证分数并未同步改善。

训练累计有效时间131.75分钟，峰值显存3642.1 MiB，参数量1,288,841。有效更新40,000×batch128=5,120,000个采样窗口，与示例一致；参数量和参数量×样本数均为示例的1.7096倍，后者为6.59887e+12。该口径不等于FLOPs或额外重放开销。

首次启动在第100步完整断点后遇到CUDA unknown error；随后从该断点恢复模型、优化器、OneCycle、RNG及批次位置，保持训练会话连接后完成。没有用D权重warm start，没有重置优化器；最终完整断点的模型、优化器和调度器均核对为40,000步。运行日志保留初次失败与恢复记录，以上训练时间为训练器记录的累计时间，不包含丢弃更新和故障排查开销。

复现从随机初始化开始，使用下方原命令和新的输出前缀；中断恢复只用本轮完整状态。该轮报告时no_cross/flat的C模型尚未运行dev；后续已按例外许可完成三组dev，目标尺度方案仍暂缓，Phase B未开始。

- checkpoint SHA256：`950787015c95233805a5f97f97ec6997c9f96d30a4f8393cce7406666542a19d`
- 最终权重：`data/uav_arena/participants/jepa/phase_a_full_e_40000_v1.pt`
- 完整断点：`data/uav_arena/participants/jepa/phase_a_full_e_40000_v1.resume.pt`
- 官方结果：`data/uav_arena/results/phase_a_full_e_40000_v1_openloop.json`
- 结果对照：`data/uav_arena/results/jepa/full_e_40000_comparison.json`

## full E从头训练与验证复现命令

依据维护者2026-10-29的[训练要求](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/3#issuecomment-6488896970)及[变体补充](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/3#issuecomment-6489347290)：

- 使用D的命令，只把steps改为40,000，seed固定为0，从随机初始化开始；不加载D权重，不修改模型、损失、骨干或归一化。
- 新输出前缀`phase_a_full_e_40000_v1`。训练完成后在完整100个验证episodes上按stride10、四种延迟评分，以原始分数严格判断`< 0.082593`。
- 样本量40,000×128=5,120,000，与门槛示例5,000×1,024完全相同；D的样本量为示例57.5%。
- 参数量1,288,841，为示例753,889的约1.71倍；参数量×样本数为6.599e12，为示例约1.71倍。Phase A检查实现能否使用，不要求算力持平。
- 如果仍不过线或变差，先报告，不换seed、不自动加训、不同时改其他配置。Phase B在2,000局上另选配置。
- 门槛按共享代码库判：full过线即可。no_cross、flat使用现有C模型，不重训，不要求各自过线。
- 后续2026-10-30例外决定替代原门槛前置条件：full使用D模型，no_cross/flat使用各自C模型，分别执行dev seeds 0–4、8种条件（各40局），只查闭环。
- 维护者复现full并审阅三组dev后决定能否进入Phase B；共享README及W-ar配对协调不阻塞本轮。

维护者示例seed0–4的分数依次为8.2593、8.2318、8.2574、8.2344、8.2603 cm，标准差约0.014 cm。D的差距不能按噪声忽略，不换seed撞线。

本轮执行的从头训练命令（复现时使用新输出名）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
data/uav_arena/.venv/bin/python -u -m uav_arena.participants.jepa.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 250 --steps 40000 \
  --batch 128 --history 1 --rollout-steps 15 --dim 128 --latent 64 \
  --enc-layers 2 --pred-layers 4 --heads 4 --attn full \
  --head-weight 20 --sigreg-weight 0.01 --sigreg-projections 1024 \
  --stats-samples 5000 --lr 0.001 --seed 0 --workers 4 --device cuda \
  --out data/uav_arena/participants/jepa/phase_a_full_e_40000_v1.pt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/phase_a_full_e_40000_v1.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --stride 10 \
  --out data/uav_arena/results/phase_a_full_e_40000_v1_openloop.json
```

完整训练状态每100步保存，恢复只能使用本轮`.resume.pt`，不会重置优化器。

## 维护者2026-10-30例外放行与Phase A收尾

[维护者评论](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/3#issuecomment-6502819652)批准这套代码带着已记录的开环差距推进：full最佳仍为D 8.3737 cm，E 8.4265 cm，精确门槛8.2593 cm，D高约0.1144 cm（约1.4%）。**数值门槛没有通过；这是有记录的例外许可，不将gate_passed改为true。**

依据：实现、数据校验和测试未发现问题，短期/中段预测与示例接近，冻结编码器读出约1.6 cm；D到E训练目标下降但验证变差，可能受250局过拟合影响；Phase B将使用2,000局并重新选配置；冻结主分析保留三个变体，许可作于JEPA dev/test之前。

当前仅获准完成诊断与dev收尾。维护者本机复现D、审阅三组dev后，正式开始Phase B；网格需先在PR审阅。PR保持Draft，合并在Phase B完成之后。README归属及与june对齐history/宽度/full mask仍需处理，不阻塞本轮。

### D/E前100训练局诊断（不选配置）

两个模型均在explore_v2的seeds 200000–200099上执行官方openloop，stride10、15步、四种延迟；与已有explore_v2_val seeds 299000–299099的验证结果并排。训练集评分不是验证选择分数，也不是U记录状态上的正式L1。

| 模型 | 训练步数 | 前100训练局cm | 完整100验证局cm | 验证−训练cm |
|---|---:|---:|---:|---:|
| D | 23,000 | 2.3991 | 8.3737 | 5.9745 |
| E | 40,000 | 1.4156 | 8.4265 | 7.0109 |

| 延迟 | D训练cm | E训练cm | D验证cm | E验证cm |
|---|---:|---:|---:|---:|
| 0ms | 2.3235 | 1.3667 | 8.0500 | 8.0847 |
| 50ms | 2.3510 | 1.3775 | 8.2423 | 8.3009 |
| 100ms | 2.3797 | 1.3832 | 8.4152 | 8.4947 |
| 200ms | 2.5423 | 1.5351 | 8.7871 | 8.8257 |

E−D：训练集-0.9835 cm，验证集+0.0528 cm。E在训练局上更好、验证集更差，支持维护者提出的过拟合判断；这是固定seed和训练子集的诊断，不单独证明全部性能差距的原因。

诊断未修改权重、归一化或源码；四个checkpoint哈希与原记录一致。既定full dev模型仍为D，不根据本次训练集评分重新选择。

复现诊断（已有输出时换用新前缀）：

```bash
for run in phase_a_full_d_fresh_v1 phase_a_full_e_40000_v1; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    data/uav_arena/.venv/bin/python -m uav_arena.openloop \
    --predictor uav_arena.participants.jepa.model:make \
    --predictor-options "{\"checkpoint\":\"data/uav_arena/participants/jepa/$run.pt\",\"device\":\"cuda\"}" \
    --data data/uav_arena/datasets/explore_v2 --seeds 200000:200100 --stride 10 \
    --out data/uav_arena/results/${run}_openloop_train100.json
done
```

### 指定dev模型（已完成）

| 变体 | 模型 | dev状态 |
|---|---|---|
| full | phase_a_full_d_fresh_v1 | 已完成，28/40成功 |
| no_cross | phase_a_no_cross_c | 已完成，8/40成功 |
| flat | phase_a_flat_c | 已完成，0/40成功 |

每个模型使用dev --seeds 0:5、全部8种条件，只查闭环，不选配置、不重训。本块新增39项本地编排测试通过，覆盖批准记录、D/C映射、保持数值门槛未通过，以及训练集诊断的seeds/stride/latency和错误结果拒绝。旧E编排的dev入口已改为读取明确的例外许可，并使用D/C模型；其E评分与gate_passed=false保持原样，例外收尾状态单独保存。

许可及诊断文件均留在data/：

- data/uav_arena/results/jepa/phase_a_closeout_approval.json
- data/uav_arena/results/jepa/phase_a_closeout_status.json
- data/uav_arena/results/jepa/phase_a_train100_diagnostic.json
- data/uav_arena/results/phase_a_full_d_fresh_v1_openloop_train100.json
- data/uav_arena/results/phase_a_full_e_40000_v1_openloop_train100.json

## 三组dev闭环检查（120局，已完成）

按维护者2026-10-30例外收尾许可，模型在dev之前固定：full使用D，no_cross/flat使用各自C。每个模型40局，dev seeds 0–4×全部8种条件；MPPI 512候选、15步horizon、默认var_scale0.04、temperature0.05、planner seed0。不是选择配置的实验，未据dev更换模型或参数。

| 变体 | 使用模型 | 成功/40 | 成功率 | 成功平均到达时间s | MPPI每决策平均ms |
|---|---|---:|---:|---:|---:|
| full | phase_a_full_d_fresh_v1 | 28/40 | 70.0% | 5.50 | 36.39 |
| no_cross | phase_a_no_cross_c | 8/40 | 20.0% | 4.82 | 36.33 |
| flat | phase_a_flat_c | 0/40 | 0.0% | — | 35.27 |

| 条件 | full D成功/5 | no_cross C成功/5 | flat C成功/5 |
|---|---:|---:|---:|
| static/0ms | 3/5 | 0/5 | 0/5 |
| static/50ms | 2/5 | 0/5 | 0/5 |
| static/100ms | 4/5 | 3/5 | 0/5 |
| static/200ms | 3/5 | 1/5 | 0/5 |
| static+moving/0ms | 3/5 | 1/5 | 0/5 |
| static+moving/50ms | 5/5 | 1/5 | 0/5 |
| static+moving/100ms | 5/5 | 2/5 | 0/5 |
| static+moving/200ms | 3/5 | 0/5 | 0/5 |

| 变体 | 失败类型与次数 |
|---|---|
| full | collision_team: 10; collision_obstacle: 2 |
| no_cross | collision_team: 24; collision_obstacle: 8 |
| flat | collision_team: 40 |

本轮46项本地编排测试通过，包含部分结果保留、只补缺失组合及完整结果不重跑检查。每个模型均核对40个唯一seed/condition组合；模型hash、预测接口和冻结规划参数匹配原记录。本轮full使用23,000步D，另外两个变体使用10,000步C；训练预算不同，dev只作闭环检查，不作为公平架构排名。collision_team表示机间距离低于环境规定阈值，不等同于确认发生机体接触。到达时间只对成功episodes平均；MPPI耗时按决策数加权，包含完整规划循环，不能与前述单次预测调用耗时直接比较。各条件完整失败类型及规划耗时留在原始JSON和本地汇总中。

三组dev复现命令（使用新的输出前缀，不覆盖原始结果）：

```bash
for run in phase_a_full_d_fresh_v1 phase_a_no_cross_c phase_a_flat_c; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
    data/uav_arena/.venv/bin/python -m uav_arena.run \
    --predictor uav_arena.participants.jepa.model:make \
    --predictor-options "{\"checkpoint\":\"data/uav_arena/participants/jepa/$run.pt\",\"device\":\"cuda\"}" \
    --suite dev --seeds 0:5 --conditions all --solver mppi --horizon 15 --samples 512 \
    --var-scale 0.04 --temperature 0.05 --seed 0 \
    --out data/uav_arena/results/reproduce_${run}_dev.json
done
```

完整结果保存在data/uav_arena/results/下：

- phase_a_full_d_fresh_v1_dev.json
- phase_a_no_cross_c_dev.json
- phase_a_flat_c_dev.json
- jepa/phase_a_dev_closeout_summary.json

数值门槛仍未通过，gate_passed=false；本轮只有例外收尾许可。D/E训练集诊断及三组dev已完成，正式Phase B仍须维护者复现D、审阅dev与4配置候选网格后启动。PR继续Draft，未运行test、未启动Phase B训练。

## 验证

相关112项测试通过、1项跳过（在106项基础上新增2项续训和4项完整断点检查），覆盖Arena回归、状态排列、15步反传、1/512/555候选、reset、
不读取载荷质量、时间因果性、坐标平移和checkpoint重载。新增检查包括冻结参数、物理单位、逐episode验证stride、训练/验证划分、独立权重保存、原始shuffle顺序和跨epoch完整恢复。

## 后续任务与验收

按[TASKS.md](../../TASKS.md)推进三条JEPA模型行，并与W-ar共享骨干。

### Phase A：训练收敛与闭环检查

- [x] 实现三个变体、共享骨干、训练入口、预测接口及测试。
- [x] 完成前250个训练episodes的七次试验，记录完整验证集的stride-10分数、参数量与计算开销。
- [x] 完成full C/D的读出探针与官方多步误差对照，记录增加训练预算的结果。
- [x] 接收维护者结论：Phase A暂缓逐步目标归一化，保留原损失配方。
- [x] full按D配置、seed0完成40,000步；验证8.4265 cm，未过线。
- [x] 报告40,000步结果；维护者2026-10-30批准带差距完成Phase A收尾，不再增加Phase A训练。
- [x] 完成D/E前100训练局stride10诊断并与已有验证分数对照。
- [x] 记录例外放行依据：full D仍未低于0.082593 m，gate_passed=false；条件许可与数值门槛分别记录，no_cross/flat不重训。
- [x] 按例外许可完成full D、no_cross C、flat C各40局dev，共120局；未用于选择配置。
- [x] 将训练集诊断、三组dev和4配置网格纳入现有草稿PR；正式Phase B等待维护者复现D并确认网格。
- [ ] 与june确认共享README归属及W-ar配对实验的history、full mask、骨干宽度和含读出头参数量；不阻塞本轮训练。

### Phase B：维护者审阅后执行

已准备[共享4配置网格](PHASE_B_GRID.md)，包含history1/4与23,000/40,000步；12次选择训练预计约19.95 GPU小时（仅训练估计）。尚未启动，待维护者复现D及审阅。

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
