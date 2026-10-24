# E1 混合候选、评分训练与新场景确认：整块执行报告

## 本批结论

连续完成三个大块：①固定协议与24新训练场景补采；②18+6训练场景内选择/24场景重拟合；③冻结后8新场景确认。得到真实非零残差模型 **ridge λ=1**。内部regret改善22.87%，新确认集改善仅6.69%，8场景中2改善/1退步/5持平，**不足以替代readout或启动闭环**。

没有在确认结果后改模型、重选超参或换场景。按结果前固定的门槛自动跳过闭环，readout继续作为当前基线；候选模型保留用于研究。该停止由科学门槛触发，不是在等待逐步人工批准。

## 三块实际产出

| 大块 | 实际工作 | 产出 |
|---|---|---|
| A 数据与协议 | train24新场景、每场景2锚点、每状态64混合候选 | 48组/3072条真实标签；输入与监督分文件 |
| B 小模型训练 | 前18场景拟合、后6场景内部选择；零修正、两种ridge、737参数MLP | 内部拟合checkpoint、history、选择记录；最终24场景refit |
| C 冻结确认与门控 | 模型hash冻结后才生成8新确认场景 | 16组/1024条标签、确认分数与逐场景指标；闭环自动skip |

合计32新场景、64状态组、4096真实标签。每组包含逐步随机16、block随机16、readout CEM最终池16、privileged接触block16，完整池内比较；各方法选同一64候选池。CEM中间2048条预测没有真实标签，另存UNLABELED，未计入4096。

新train seed2026102900–2026102923；新confirmation seed2026103000–2026103007。与旧ranking_v2和phase21场景manifest核对seed/initial state，新train/confirmation间同样核对无重合；这不证明与未知world-model预训练数据无重合。旧test候选NPZ与test_evaluation本轮打开0。确认集本轮已经使用，后续必须视为已消耗数据，不可反复挑参数。

## 固定目标和模型

主目标仍是原physical global cost：norm([agent0xy error,Txy error])+角差权重。T-only改善只作为分项诊断及防退步约束，没有替换原任务/成功标准。

21维规划特征只来自冻结probe的预测pose/图像goal、动作均值/标准差和预测T位移/角变化。没有真实状态、真实接触、proposal标签或真实未来进入模型特征。接触候选本身来自真实几何，源状态来自CoopPolicy，因此整个实验包含privileged候选和源分布限制，不能称纯图像在线planner。

ridge拟合组内中心化的(true−readout)/100残差，修正限制±25真实代价单位。MLP固定21→32→1，零初始化输出，tanh限幅；单训练seed0、最多100epoch、patience20，pairwise softplus+0.1 MSE。normalize仅拟合集；内部选择完后用全部24训练场景重算统计/refit，确认不参与。

## 内部6场景选择

| 候选 | regret | PRA | 所选T-only gain | 达到内部资格 |
|---|---:|---:|---:|---|
| zero_correction | 8.5919 | 88.28% | 9.2379 | 是 |
| ridge_lambda_1 | 6.6266 | 89.19% | 12.2803 | 是 |
| ridge_lambda_0.01 | 8.2307 | 88.21% | 8.0217 | 否 |
| residual_mlp_21_32_1 | 6.6822 | 88.45% | 11.5071 | 是 |

内部资格预先要求regret至少相对改善5%，且平均T-only gain不低于readout超过1单位；随后按global regret选择，平局优先基线/强正则/小模型。选择ridge λ=1，然后只在24训练场景refit该模型。

## 新确认8场景结果

| 指标 | readout | 固定残差模型 |
|---|---:|---:|
| 场景平均global regret ↓ | 5.865985 | 5.473765 |
| PRA ↑ | 84.6462% | 84.6436% |
| 所选T-only gain ↑ | 18.905932 | 18.898969 |
| 所选global gain ↑ | 54.012041 | 54.404261 |

| scene seed | readout regret | 残差regret | 残差−readout |
|---|---:|---:|---:|
| 2026103000 | 0.8800 | 0.1764 | -0.7037 |
| 2026103001 | 1.8960 | 1.8960 | 0.0000 |
| 2026103002 | 15.3464 | 15.3464 | 0.0000 |
| 2026103003 | 4.3041 | 4.6163 | 0.3122 |
| 2026103004 | 2.7927 | 2.7927 | 0.0000 |
| 2026103005 | 7.1420 | 4.3957 | -2.7463 |
| 2026103006 | 11.7149 | 11.7149 | 0.0000 |
| 2026103007 | 2.8517 | 2.8517 | 0.0000 |

按场景内两状态平均，再8场景等权平均；候选数不是独立样本量。只作小样本描述性确认，不给显著改善结论。T-only gain大致不变，PRA也未改善，说明新模型获益有限且未稳定覆盖多数场景。

预先闭环门槛：非零模型；确认regret改善≥10%；≥5/8场景获益；T-only不退步超过1。实际6.69%且2/8，两个条件不满足；closed_loop/summary.json明确status=skipped。闭环详细协议已在确认结果产生前冻结，未运行任何闭环episode，不报告成功率。

## 验证、算力和失败边界

全部4096候选均reset(seed)+完整prefix后分支，并逐候选第二遍native replay，state/body误差0。独立数组复核：标签重算、pose、21维特征重建、动作范围、block连续性、接触支持、分文件隔离、原权重hash全部通过。train与confirmation特征/代价重算最大误差均0。

最终ridge在train上独立重建正规方程，beta最大差3.56e-17；train-only mean/std逐元素匹配。Windows NumPy从portable checkpoint独立重算全部确认分数，最大误差0.00e+00，逐场景regret和gate复现。没有降检查标准、跳过失败场景或改原始权重。

训练数据采集128.29秒，确认采集50.67秒，训练选择/重拟合0.44秒；均不含脚本开发和工具启动。RTX4060用于冻结模型推理，训练小模型CPU完成；实际学习模型候选评分共7168条次。采集期PyTorch峰值allocated约128.2MiB，不代表整机GPU峰值或最低显存。

首次ready缺失按预定t75回退：train1场景、确认0场景，未替换。没有新增生产依赖、没有提交或推送、没有下载新模型权重。

## 后续按大块执行的安排

1. **预测与读出诊断块**：固定训练数据，分别分析真实帧probe误差、预测latent到真实编码latent的差距、接触/非接触及T分项；据证据决定predicted-latent readout适配或阶段评分，不继续盲目扩网。
2. **单项修复与确认块**：每次只干预一个环节，训练场景内部选择，冻结后另设新确认场景；本轮确认集不再用作选择依据。只有超过同池readout且T分项不退步，才进入少量闭环。
3. **Navigation接入块**：收到组员E4仓库/模型后，一次完成环境、schema、checkpoint、readout与评估协议核查，再接现有E1组件；材料缺失会直接列给用户。本地PushT权重不代替Navigation模型。

后续默认按“诊断→实现→训练/评估→验证→归档”完整块推进；仅在缺关键材料、不可逆操作或超出已确定预算时请求输入。本批已到固定停止门槛，当前没有后台任务。

## 文件位置

protocol.json/closed_loop_protocol.json：预设协议；train_data/confirmation_data：原始数据与metadata/manifest；selected_scorer.pt和selected_scorer_portable.npz：候选模型；inner_fitted_models.pt/training_history.json/selection_frozen.json：训练审计；confirmation_scores.npz/confirmation_group_metrics.json/confirmation_summary.json：已冻结确认；各verification.json：验证。源代码归档scripts/phase22_v1。原权威Multi_Agent_LeWM.md未修改。
