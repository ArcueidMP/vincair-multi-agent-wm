# full Phase B 四配置选择完成

核对时间：2026-11-01T09:16:37.768351+08:00。full G1–G4训练与完整验证均完成，队列正常退出。

| 配置 | history | steps | selection score cm | 训练分钟 |
|---|---:|---:|---:|---:|
| G1 | 1 | 23,000 | 9.206615 | 77.48 |
| G2 | 1 | 40,000 | 7.923732 | 132.10 |
| G3 | 4 | 23,000 | 7.166845 | 89.10 |
| G4 | 4 | 40,000 | 5.918571 | 154.85 |

当前四配置最优为 **G4：history4、40,000步，5.918571 cm**，参数1,289,225；相比同history的G3降低1.248274 cm（17.42%）。全部训练累计7.5589 GPU小时，不含完整验证与数据统计。

## 验证口径与证据

每次均用全部2,000训练局、seed0；验证全部100局（299000–299099），四延迟各7,429决策点、15步、stride1，只评最终checkpoint。原始selection_score以m保存，表格换算为cm。已核对最终训练步数、完整断点的优化器与调度器步数、最终权重对应关系、权重与结果SHA256及验证覆盖；源码与数据身份保持一致。

分延迟载荷误差（cm）：

| 配置 | 0 ms | 50 ms | 100 ms | 200 ms |
|---|---:|---:|---:|---:|
| G1 | 8.907943 | 9.089280 | 9.258855 | 9.570381 |
| G2 | 7.681412 | 7.816548 | 7.967038 | 8.229928 |
| G3 | 6.891867 | 7.034317 | 7.214598 | 7.526596 |
| G4 | 5.608173 | 5.745389 | 6.000727 | 6.319994 |

checkpoint SHA256：

- G1: `c1dbee9b4438f240c6109d42c21e21e1a7bc2e380646f151ad95fe4a5a93d2cc`
- G2: `237823eebf70b79e21515617d7adbc028daa286072af867e9fc2752a61222f2d`
- G3: `d35ce6793d55654c36555ec68ebf05619b70bb1ad3161a2652c930ada695abd0`
- G4: `cb84b1ff80f2a7f18b7417e7738b98bb07be9364317cffb3c9edce82277501dc`

## 需要维护者决定

40,000步在两个history下均优于23,000步。按评论6520010790先报告，由维护者判断是否在最多8配置以内给三个变体共同增加更长步数。本次报告full四配置结果并提请审阅；未自行扩展网格或开始正式新seed模型，未运行新的dev/test。

当前网格选中的history为4；W-ar的Phase B应采用相同history和full骨干。若获批扩展改变选择，须同步更新。Phase A的gate_passed=false历史记录保留。


## W-ar训练交接：history=4

W-ar现由维护者训练，依据[PR #2最新分工](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/2#issuecomment-6554817446)。full四配置结果在此提供给W-ar选型，不再等待no_cross/flat结束才披露。

| 骨干选项 | 当前full G4 |
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

full G4固定配方：batch128、lr0.001、head_weight20、SIGReg0.01、projections1024、rollout15、stats_samples5000、AdamW/OneCycle，选择seed0。W-ar不需要照搬JEPA专属损失。

## 复现命令

在仓库根目录执行以下命令可从头复现G4；已有同名结果时不要覆盖，请先选用独立输出前缀。三个变体的完整批准网格与队列说明见[PHASE_B_GRID.md](PHASE_B_GRID.md)。

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.train \
  --data data/uav_arena/datasets/explore_v2 --episodes 2000 \
  --history 4 --steps 40000 --attn full --batch 128 --lr 0.001 \
  --dim 128 --latent 64 --enc-layers 2 --pred-layers 4 --heads 4 \
  --head-weight 20 --sigreg-weight 0.01 --sigreg-projections 1024 \
  --rollout-steps 15 --stats-samples 5000 --seed 0 --workers 4 --device cuda \
  --out data/uav_arena/participants/jepa/phase_b_select_full_g4_s0.pt

PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/phase_b_select_full_g4_s0.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --seeds 299000:299100 \
  --out data/uav_arena/results/phase_b_select_full_g4_s0_openloop.json
```

G1/G2/G3分别把history/steps改为1/23000、1/40000、4/23000，同时使用各自输出名前缀；其他参数保持一致。完整验证不传--stride，默认1。

## 结果身份与检查

原始JSON、权重、完整断点与日志留在data/，不加入Git：

- `data/uav_arena/results/jepa/phase_b_full_selection.json`：四配置与选择结果。
- `data/uav_arena/results/jepa/phase_b_select_full_g{1..4}_s0_queue.json`：checkpoint与result SHA256绑定。
- `data/uav_arena/results/phase_b_select_full_g{1..4}_s0_openloop.json`：完整验证分项。
- `data/uav_arena/participants/jepa/phase_b_select_full_g{1..4}_s0.pt`及`.resume.pt`：最终权重与优化器/调度器/RNG状态。
- `data/uav_arena/logs/phase_b_select_full_g{1..4}_s0_{train,val}.log`：独立日志。

标准数据校验：训练match2000/differ0/unknown0，验证match100/differ0/unknown0。数据manifest SHA256：

- explore_v2：`cb196ded5be80b12039fc65e1c3797ab50113394046a7f03b8c1de97ac7959c0`
- explore_v2_val：`2e6a5c8fd32fcbff297110248570e78c7f4ec6dd557be33fa383555312e8cab0`

以上为完整验证的选型结果，不作为正式L1/L2/L3/J主分析数据。

## 本次PR更新验证

骨干、参赛接口与队列测试59通过，完整断点测试4通过，共63通过；仅在CPU执行测试，未重新训练或评分。git diff --check通过。冻结Arena文件、骨干、训练器、损失与模型源码未修改。

结果JSON SHA256：

- G1: `b87dc7d3c1b2a98a8462fd4c8d7b6d003ca6aadc9d21be28fa2367558d873e89`
- G2: `212f49050d98f61ec90be734b626611afd56a6e2ad68001c35fcd8bd3ed78986`
- G3: `b9474d167da25bfb4fbdf9287d247401319c9f08df841ea4dd64d30ebad88aff`
- G4: `cad3e4f4375c310b9334598bbfab1e4ea8168a2f1e42392054bb608cd8fd4915`
