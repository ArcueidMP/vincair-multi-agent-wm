# Phase B配置网格（G1–G6已完成；最终均选G6）

Phase A收尾已完成：D/E训练集诊断与三组各40局dev结果见[PHASE_A.md](PHASE_A.md)。维护者2026-10-30已通过收尾并批准四配置网格（评论6520010790）；复现D延后，不阻塞训练。先完成full G1–G4选择（评论6522103322），供W-ar确定history。PR保持Draft，合并放在Phase B完成后。

维护者已在[评论6555210796](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/3#issuecomment-6555210796)批准三个变体共同新增G5/G6；60,000步为最终上限，即使仍在改善也不再扩展。三个变体六配置训练与完整验证已全部完成，最终均选G6；完整分数、耗时及哈希见[PHASE_B_SELECTION.md](PHASE_B_SELECTION.md)。

## 共享的6配置

| 配置 | history | steps | full/no_cross参数 | flat参数 | 实测训练GPU小时：full / no_cross / flat |
|---|---:|---:|---:|---:|---|
| G1 | 1 | 23,000 | 1,288,841 | 1,371,663 | 1.29 / 1.26 / 0.82 |
| G2 | 1 | 40,000 | 1,288,841 | 1,371,663 | 2.20 / 2.28 / 1.29 |
| G3 | 4 | 23,000 | 1,289,225 | 1,372,047 | 1.48 / 1.37 / 0.85 |
| G4 | 4 | 40,000 | 1,289,225 | 1,372,047 | 2.58 / 2.45 / 1.53 |
| G5 | 1 | 60,000 | 1,288,841 | 1,371,663 | 3.52 / 3.16 / 1.98 |
| G6 | 4 | 60,000 | 1,289,225 | 1,372,047 | 3.75 / 3.77 / 2.28 |

三个变体共用相同网格，各自选优，共18次选择训练。G1–G4共12次核验后复用，G5/G6共6次补跑已完成。只改变history与steps，保留C/D配方：batch128、lr0.001、head_weight20、SIGReg0.01、projections1024、rollout15、dim128、latent64、encoder2、predictor4、heads4、dropout0、stats_samples5000、AdamW/OneCycle、seed0。full/no_cross每帧4×64 latent，flat每帧256，参数量按实际模型构造统计。

每次从随机初始化开始，用explore_v2全部2,000局（seeds200000–201999），归一化仍只由训练数据统计。选择运行的seed固定为0；不会复用Phase A权重。只评最终checkpoint，不用中间checkpoint、dev或test选配置。

## 完整验证与选择

每个候选均使用explore_v2_val全部100局（299000–299099）、四延迟、15步、每个decision的官方openloop。**命令不传--stride，使用默认1；不得沿用Phase A的stride10。** 原始selection_score越低越好，各变体独立选最优；若原始分数完全相同，按G1→G6编号顺序选，不按四舍五入分数判同分。

所有候选配置、结果、权重哈希和失败运行均保留；基础设施中断可恢复同一实验完整状态，不更换seed或修改配置。候选输出名使用phase_b_select_变体_g1至g6_s0，现有检查未发现路径冲突。

原审阅草案保留在`phase_b_grid_draft.json`；当前批准元数据和命令保存在`data/uav_arena/results/jepa/phase_b_grid.json`，不会由本次报告自动执行。以下仅示意G1 full的完整验证口径：

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.openloop \
  --predictor uav_arena.participants.jepa.model:make \
  --predictor-options '{"checkpoint":"data/uav_arena/participants/jepa/phase_b_select_full_g1_s0.pt","device":"cuda"}' \
  --data data/uav_arena/datasets/explore_v2_val --seeds 299000:299100 \
  --out data/uav_arena/results/phase_b_select_full_g1_s0_openloop.json
```

## 计算开销

全部18次选择训练实测累计37.8768 GPU小时，其中新增G5/G6共18.4707 GPU小时，接近维护者约19小时的估计。表格已替换为训练元数据记录的实测耗时，不含验证、数据加载及统计。正式多seed训练未计入。

## W-ar / PETS对齐信息

W-ar沿用同一tokenizer/transformer，full mask、history及宽度与JEPA full配对。骨干接口不变，默认history4；候选history1/4需要在W-ar CLI与backbone-options中保持一致。宽度固定dim128、latent64、encoder2、predictor4、heads4。JEPA的读出为规划所需15维，W-ar为51维状态，需将各自读出头计入总参数并尽量匹配，不能只说骨干相同。

| history | full骨干参数 | JEPA读出参数 | full含读出总参数目标 | flat含读出总参数参考 |
|---|---:|---:|---:|---:|
| 1 | 1,271,040 | 17,801 | 1,288,841 | 1,371,663 |
| 4 | 1,271,424 | 17,801 | 1,289,225 | 1,372,047 |

PETS总容量按transformer最大行确定，本表保留full与flat参数供对照。维护者已在PR #2评论6522101172将PETS五成员总参数上限固定为1,372,047（包括log-variance上下界）；W-ar与full的7,860参数差异可接受，无须补参数。W-ar骨干、full mask与最终history跟随full选型。

共享README归属仍待协商。上述信息随PR公开给协作者查看；未额外发送群聊或私信。

## 后续正式实验边界

候选选优后，每变体固定配置，在2,000/500局各训练3个新seed，两种数据预算保持相同步数，共18个正式JEPA模型；正式新seed按最新审阅采用1、2、3，三个变体与两种数据量配对。test及L1/L2/L3/J按冻结ANALYSIS.md执行，所有变体保留，不按Phase A dev成功率筛掉变体。

**三个变体最终均选G6（history=4、60,000步）**：full 5.069164 cm、no_cross 4.671645 cm、flat 6.559518 cm。全部六配置完整验证及哈希核验已完成；结果见[PHASE_B_SELECTION.md](PHASE_B_SELECTION.md)。W-ar最终history保持4，骨干交接见[PHASE_B_FULL.md](PHASE_B_FULL.md)。

### 2026-11-02归档与迁移

旧网格、三份四配置汇总、12次训练的最终权重/完整断点/训练元数据、验证结果、队列状态、train/val日志及迁移前源码文档已复制到`data/uav_arena/results/jepa/archive/phase_b_g1_g4_2026-11-02/`。`manifest.json`逐文件记录路径、大小与SHA256；99个文件全部核验一致，原文件保留原位。历史block3状态和旧汇总中的扩展待审标志保留于归档，不代表批准仍待定。

迁移时活动selection汇总曾保留四配置暂定结果；2026-11-03队列完成后已更新为六配置最终汇总，`selection_complete=true`、`selected_is_provisional=false`，历史四配置汇总保留在归档。G5/G6使用独立输出名、seed0随机初始化及完整60,000步OneCycle计划，不能从G2/G4的40,000步断点接着训；同一G5/G6实验若中断，可恢复自身完整断点。训练器、骨干、损失、归一化和数据均未改变。


## 串行训练与验证队列

队列默认只检查，不启动训练：

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.selection_queue --variant full
# 显式执行；已完成候选核验后复用：
PYTHONPATH=. data/uav_arena/.venv/bin/python -m uav_arena.participants.jepa.selection_queue --variant full --run
```

`--variant no_cross`或`flat`各处理该变体G1–G6，后续补跑顺序为full G5/G6→no_cross G5/G6→flat G5/G6；现有G1–G4只核验和复用。每个候选从头训练后立即完整验证；中断时只恢复该候选完整断点。完成结果经配置、最终步数、源码/数据/权重哈希及验证覆盖检查后跳过；异常停止，不重置优化器、不自动改配置，不覆盖不完整或不匹配输出。必须保留前台WSL会话。

队列不运行dev/test，不发布PR。训练、验证日志在`data/uav_arena/logs/`，每个候选队列状态和变体选择汇总在`data/uav_arena/results/jepa/`。汇总记录六配置完整性与60,000步硬上限，不再标记需要扩展；不自动开始正式多seed训练。


## 新checkout准备队列元数据

`data/`不随Git提交。复制标准数据并按PARTICIPANTS.md完成collect --verify后，可在仓库根目录生成队列所需的批准网格元数据。该命令不训练、不覆盖现有文件；flat的clearance依据PHASE_A.md中的顺序检查和本PR测试。

```bash
PYTHONPATH=. data/uav_arena/.venv/bin/python - <<'PY'
from pathlib import Path
from uav_arena.participants.jepa.selection_queue import COMMON, CONFIGS, digest
import json

root = Path('.')
docs = root/'uav_arena/participants/jepa'
grid = dict(status='approved_ready', common=COMMON, training_episodes=2000,
            extension_approval_url='https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/3#issuecomment-6555210796',
            hard_step_cap=60000,
            validation_stride=1, selection_seed=0, horizon=15,
            validation_decisions_per_latency=7429, flat_preflight={'cleared': True},
            source_sha256={n: digest(docs/n) for n in
                ('train.py', 'objective.py', 'model.py', 'backbone.py', 'training_checkpoint.py')},
            data_manifest_sha256={n: digest(root/'data/uav_arena/datasets'/n/'manifest.json')
                for n in ('explore_v2', 'explore_v2_val')}, configs=[])
for label, history, steps in CONFIGS:
    variants = {}
    for variant in ('full', 'no_cross', 'flat'):
        name = f'phase_b_select_{variant}_{label.lower()}_s0'
        total = ({1: 1371663, 4: 1372047} if variant == 'flat'
                 else {1: 1288841, 4: 1289225})[history]
        variants[variant] = dict(parameters={'total': total},
            checkpoint=f'data/uav_arena/participants/jepa/{name}.pt',
            result=f'data/uav_arena/results/{name}_openloop.json')
    grid['configs'].append(dict(id=label, history=history, steps=steps, variants=variants))
path = root/'data/uav_arena/results/jepa/phase_b_grid.json'
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x') as stream:
    json.dump(grid, stream, indent=2)
    stream.write('\n')
PY
```

已有队列需保留原元数据、数据manifest、源码及状态；不要用该初始化命令替换正在运行或已完成的实验。新checkout的首次完整复现应使用独立输出目录或空的data结果目录。
