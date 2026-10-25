# Navigation 训练执行补充计划（2026-10-25）

原始`Multi_Agent_LeWM.md`保持不变。本补充根据组员最新回复调整实施顺序，不替代原研究目标，也不把聊天经验当正式实验协议。

## 方向调整

- 主线从“等现成Navigation权重”转为“核对协议、自己采集和训练数值观测Entity-Token JEPA/LeWM-style模型”。组员可并行重训，之后交换模型/配置/指标。
- 组员称现有Navigation checkpoint为旧的非entity版本、不匹配当前架构，建议按相同采集协议自行训练，并尽量对齐原VMAS MARL基线和LeWM配置。具体脚本/commit尚未收到。
- 组员给出的30–45分钟CPU采集、3–4小时laptop GPU训练是经验值，数据量、配置和计时口径未知；正式预算以本机剖析和实际协议为准。

## 已执行的第一块

1. 审计发现旧`vmas_pusht`的T字段、归一化、token和head不可直接用于Navigation；新建独立模块，原PushT代码和权重不覆盖。
2. 建立明确标注的本地协议`LOCAL_NAV_NUMERIC_PILOT_V1`：双机、原生18维观测中每机位置/速度4维作为集中式token输入，保存完整观测用于后续对齐。
3. 采集64集、8192 transitions，按16个完整几何拆为12 train/4 validation；head同时监督encoded和predicted latent；完成500步小模型训练与保存/重载。
4. 独立验证物理误差、时间对齐、训练统计、CPU上下文推理与原生回放。详见[试验报告](../experiments/e4_navigation/navigation_training_results/v1/REPORT.md)。

**结果：训练工程闭环通过，位置预测资格失败。** 暂不接入planner，不因已有checkpoint就宣称模型可用，也不直接投入3–4小时正式训练。

## 后续大块顺序

| 顺序 | 工作 | 进入下一块条件 |
|---|---|---|
| A | 与组员核对collector、场景、观测/动作、history/frameskip、模型commit；记录差异 | 配置来源清楚；未对齐项保留为本地实验 |
| B | 固定预算检查状态头重建、训练充分性、latent/物理多步误差；保持数据分组 | 至少优于persistence；与常速/解析对照差距和失败清楚 |
| C | 按确认配置采集正式数据并训练；增加周期checkpoint/恢复测试 | 训练数据覆盖和显存/时间预算明确，验证质量稳定 |
| D | 冻结模型后做E1候选评分和E4闭环对照 | 相同物理预测时长和候选预算，使用NEW确认几何 |
| E | 验证三机与更多agent支持 | 不能只靠API形状或理论共享参数宣称泛化；需新数据与独立评估 |

现有解析/直接评分器、真仿真基线及高惯性失败完整保留。训练集可覆盖NEW近冲突/高速几何，但不得把已消耗的旧压力集反过来当训练/新确认数据。

## 仍需组员提供的最小协议材料

优先要**当前采集脚本和config/commit**，不必等训练结束：环境/agent/goal/观测字段，策略混合与数据规模，dt/frameskip/history/action格式，网络配置/损失/归一化，数据划分和state head/readout训练方法。旧非entity权重只能做兼容性参照，不能冒充新架构权重。

本地pilot已有数值Navigation checkpoint，但**队友当前架构匹配材料仍未提供**；两者需在团队沟通和成果说明中区分。
