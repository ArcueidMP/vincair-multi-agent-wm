# Phase B配置网格草案（待维护者确认，尚未启动）

Phase A收尾已完成：D/E训练集诊断与三组各40局dev结果见[PHASE_A.md](PHASE_A.md)。维护者2026-10-30批准带开环差距推进，但要求先复现D、审阅dev和网格，再正式开跑。PR保持Draft，合并放在Phase B完成后。

## 共享的4配置

| 配置 | history | steps | full/no_cross参数 | flat参数 | 训练GPU小时：full / no_cross / flat |
|---|---:|---:|---:|---:|---|
| G1 | 1 | 23,000 | 1,288,841 | 1,371,663 | 1.26 / 1.27 / 0.78 |
| G2 | 1 | 40,000 | 1,288,841 | 1,371,663 | 2.19 / 2.20 / 1.36 |
| G3 | 4 | 23,000 | 1,289,225 | 1,372,047 | 1.51 / 1.52 / 0.94 |
| G4 | 4 | 40,000 | 1,289,225 | 1,372,047 | 2.63 / 2.64 / 1.64 |

三个变体共用相同网格，各自选优，共12次选择训练。只改变history与steps，保留C/D配方：batch128、lr0.001、head_weight20、SIGReg0.01、projections1024、rollout15、dim128、latent64、encoder2、predictor4、heads4、dropout0、stats_samples5000、AdamW/OneCycle、seed0。full/no_cross每帧4×64 latent，flat每帧256，参数量按实际模型构造统计。

每次从随机初始化开始，用explore_v2全部2,000局（seeds200000–201999），归一化仍只由训练数据统计。选择运行的seed固定为0；不会复用Phase A权重。只评最终checkpoint，不用中间checkpoint、dev或test选配置。

## 完整验证与选择

每个候选均使用explore_v2_val全部100局（299000–299099）、四延迟、15步、每个decision的官方openloop。**命令不传--stride，使用默认1；不得沿用Phase A的stride10。** 原始selection_score越低越好，各变体独立选最优；若原始分数完全相同，按G1→G4顺序选，不按四舍五入分数判同分。

所有候选配置、结果、权重哈希和失败运行均保留；基础设施中断可恢复同一实验完整状态，不更换seed或修改配置。候选输出名使用phase_b_select_变体_g1至g4_s0，现有检查未发现路径冲突。

训练与验证命令作为草案保存在`data/uav_arena/results/jepa/phase_b_grid_draft.json`，不会由本次报告自动执行。以下仅示意G1 full的完整验证口径：

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/phase_b_select_full_g1_s0.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --seeds 299000:299100 \
  --out data/uav_arena/results/phase_b_select_full_g1_s0_openloop.json
```

## 计算量估计

12次选择训练合计约 **19.95 GPU小时**，不含完整验证、2,000局加载/统计和后续正式多seed实验。history1按本机已有Phase A各变体每步耗时线性外推，full使用D、no_cross/flat使用C；history4先按同变体history1的1.20倍估计，该倍率尚未实测。模型参数量已实际统计，但时间不是2,000局上的实测，不把估计当作完成结果。

正式开跑前先确认GPU/内存和数据校验，首次运行后校准耗时；不因速度或dev结果私自更改已批准网格。若内存不足或预计成本明显变化，先报告。

## 与june对齐的信息

W-ar沿用同一tokenizer/transformer，full mask、history及宽度与JEPA full配对。骨干接口不变，默认history4；候选history1/4需要在W-ar CLI与backbone-options中保持一致。宽度固定dim128、latent64、encoder2、predictor4、heads4。JEPA的读出为规划所需15维，W-ar为51维状态，需将各自读出头计入总参数并尽量匹配，不能只说骨干相同。

| history | full骨干参数 | JEPA读出参数 | full含读出总参数目标 | flat含读出总参数参考 |
|---|---:|---:|---:|---:|
| 1 | 1,271,040 | 17,801 | 1,288,841 | 1,371,663 |
| 4 | 1,271,424 | 17,801 | 1,289,225 | 1,372,047 |

PETS按任务卡将参数上限参照transformer行：本表同时列出full与flat的总容量，供维护者/june确定采用哪一行及ensemble总参数统计口径。本稿不自行规定一个新的PETS上限，也不声称W-ar/PETS已经完成容量对齐。最终history、mask和容量上限确认后再同步执行。

共享README归属仍待协商。上述信息随PR公开给协作者查看；未额外发送群聊或私信。

## 后续正式实验边界

候选选优后，每变体固定配置，在2,000/500局各训练3个新seed，两种数据预算保持相同步数，共18个正式JEPA模型；新seed清单须先与共享协议对齐。test及L1/L2/L3/J按冻结ANALYSIS.md执行，所有变体保留，不按Phase A dev成功率筛掉变体。

**当前只提交网格审阅，没有启动任何Phase B训练。**
