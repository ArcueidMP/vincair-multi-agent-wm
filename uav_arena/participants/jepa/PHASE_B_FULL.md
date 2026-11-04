# full Phase B 六配置选择完成

2026-11-03：G1–G6训练与完整验证均完成，最终选择 **G6：history=4、60,000步，selection_score=5.069164 cm**。三个变体完整结果与哈希见[PHASE_B_SELECTION.md](PHASE_B_SELECTION.md)。

| 配置 | history | steps | full (cm) |
|---|---:|---:|---:|
| G1 | 1 | 23,000 | 9.206615 |
| G2 | 1 | 40,000 | 7.923732 |
| G3 | 4 | 23,000 | 7.166845 |
| G4 | 4 | 40,000 | 5.918571 |
| G5 | 1 | 60,000 | 6.996445 |
| G6 | 4 | 60,000 | 5.069164 |

验证使用explore_v2_val全部100局（299000–299099），四延迟0/50/100/200 ms，每延迟7,429决策，15步、stride1，只评最终checkpoint。selection_score以m保存，此处换算cm。全部候选只用explore_v2的2,000训练局、选择seed0，从随机初始化开始。

已核对每个候选的最终步数、完整断点中优化器/调度器步数、最终权重与断点模型一致性、源码及数据manifest身份、checkpoint/result SHA256绑定和完整验证覆盖。原始JSON、日志、权重及完整断点保留在data/，表格与哈希进入文档。数据校验记录：训练match2000/differ0/unknown0，验证match100/differ0/unknown0；本次重新核对manifest与训练元数据绑定。

## 分延迟载荷误差（cm）

| 配置 | 0 ms | 50 ms | 100 ms | 200 ms |
|---|---:|---:|---:|---:|
| G1 | 8.907943 | 9.089280 | 9.258855 | 9.570381 |
| G2 | 7.681412 | 7.816548 | 7.967038 | 8.229928 |
| G3 | 6.891867 | 7.034317 | 7.214598 | 7.526596 |
| G4 | 5.608173 | 5.745389 | 6.000727 | 6.319994 |
| G5 | 6.745404 | 6.862633 | 7.072005 | 7.305739 |
| G6 | 4.703157 | 4.882942 | 5.159875 | 5.530682 |

G6较同history的G4改善0.849406 cm（约14.35%）。G5/G6已按批准网格完成，60,000步为最终上限；不再扩展。正式18个新seed模型尚未启动。

## W-ar训练交接：history=4

W-ar现由维护者训练，依据[PR #2最新分工](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/2#issuecomment-6554817446)。full六配置选择已完成，最终history=4；W-ar按此对接。

| 骨干选项 | 最终full G6 |
|---|---|
| factory | `uav_arena.participants.jepa.backbone:make_backbone` |
| history | **4**；W-ar CLI与backbone-options都传4 |
| attn | `full` |
| dim / latent | 128 / 64（每实体latent宽度） |
| encoder / predictor / heads | 2 / 4 / 4 |
| dropout | 0 |
| 训练数据 | explore_v2全部2,000局；正式减量模型使用前500局 |
| 正式新seed | 1、2、3；两个预算配对 |

```bash
--backbone uav_arena.participants.jepa.backbone:make_backbone \
--history 4 \
--backbone-options '{"history":4,"attn":"full","dim":128,"latent":64,"enc_layers":2,"pred_layers":4,"heads":4}'
```

JEPA full H4骨干1,271,424参数，读出17,801，总计1,289,225；W-ar H4含状态头总计1,281,365，差7,860（约0.6%），已获[维护者接受](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/2#issuecomment-6522101172)，无须补参数。共享的是骨干实现和结构选项，不共享已训练权重。W-ar自己的batch、学习率和步数仍按其获批网格选；history与骨干不纳入W-ar网格。

full G6固定配方：batch128、lr0.001、head_weight20、SIGReg0.01、projections1024、rollout15、stats_samples5000、AdamW/OneCycle，选择seed0。W-ar不需要照搬JEPA专属损失。

## 复现命令

在仓库根目录执行以下命令可从头复现G6；已有同名结果时不要覆盖，请先选用独立输出前缀。三个变体的完整批准网格与队列说明见[PHASE_B_GRID.md](PHASE_B_GRID.md)。

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 2000 \
  --history 4 --steps 60000 --attn full --batch 128 --lr 0.001 \
  --dim 128 --latent 64 --enc-layers 2 --pred-layers 4 --heads 4 \
  --head-weight 20 --sigreg-weight 0.01 --sigreg-projections 1024 \
  --rollout-steps 15 --stats-samples 5000 --seed 0 --workers 4 --device cuda \
  --out data/uav_arena/participants/jepa/phase_b_select_full_g6_s0.pt

PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/phase_b_select_full_g6_s0.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --seeds 299000:299100 \
  --out data/uav_arena/results/phase_b_select_full_g6_s0_openloop.json
```

G1/G2/G3/G4/G5分别使用history/steps为1/23000、1/40000、4/23000、4/40000、1/60000，同时使用各自输出名前缀；其他参数保持一致。完整验证不传--stride，默认1。
