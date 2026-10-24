# Multi-Agent LeWM 个人阶段工作说明

**日期：2026-10-25。** 本文整理截至本轮的实验实施、工程贡献、主要结果和未完成事项，可用于组会说明和队友交接。原研究计划仍以用户的`Multi_Agent_LeWM.md`为准。

## 一、工作定位与贡献边界

我的工作主要是在已有 MultiPushT/VMAS 模型与队友 E4 基线上，建立可追溯的实验流程，完成复现、评分/规划失败诊断、轻量评分器训练、闭环对照和协作复现工具。

- 原始LeWM、MultiPushT/VMAS环境、作者发布模型与readout来自原项目/模型负责人，不算作我的原创模型。
- E4 notebook与早期Navigation基线来自队友分支`feature/e4-navigation-baselines`（基准提交`a4ff326`）；我完成后续脚本化复现、原生子步安全口径、数据、直接评分器、诊断与对照。
- 后续代码与实验由我在Codex辅助下实施。结果记录包含正例、失败实验及其限制，没有把直接评分器或解析动力学改称Navigation LeWM。
- 当前尚未获得Navigation专用LeWM材料，也未完成三机实验。单智能体`r-lewm-uav`、PushT模型与Navigation任务严格分开。

## 二、已经完成的工作

### 1. 运行环境与协议审计

在WSL2 Ubuntu / RTX4060 Laptop8GB上建立独立实验环境，完成权重加载、配置/动作/任务成功定义检查和运行资源记录。处理了早期存储不足问题并完成用户授权的WSL迁移。保留原项目、缓存和失败尝试。

对MultiPushT发现并记录动作边界和物理快照恢复问题。后续精确真仿真分支采用全新reset+已执行动作历史重放，而不是将不完整刚体快照冒充精确恢复。作者原协议与本地修正协议分开说明。

### 2. E0：基础复现和readout核验

- VMAS PushT完成小样本E0诊断；其checkpoint不能用于Navigation。
- MultiPushT真仿真E0-A完成15个must-move场景pilot，成功0/15；同15场景的E0-D也是0/15。它是非均衡小样本，不能写成完整95集真仿真基线。
- MultiPushT正式readout到位后完成检查及低预算两场景A/B/C smoke，不能与完整预算15集混合统计。readout拟合于真实编码帧，不等于预测latent上的充分验证。
- E0整体未完成：完整95集真仿真和完整预算B/C仍不能标为完成。

证据：[E0状态](../experiments/historical_results/e0/PHASE3_E0_STATUS.md)、[15集报告](../experiments/historical_results/e0/multipusht_e0a_15/E0A_15_REPORT.md)、[readout smoke](../experiments/historical_results/e0/readout_smoke_v1/READOUT_SMOKE_REPORT.md)。

### 3. E1：MultiPushT候选质量和评分研究

完成搜索/接触覆盖诊断、readout残差与混合候选研究。最终混合候选实验含24训练+8确认场景、64状态组、4096条真实标签，训练/确认分组与输入/未来监督分离。

固定ridge残差在内部选择中改善regret约22.87%，但新确认仅改善6.69%，8场景中2改善/1退步/5持平，未通过预设替代readout/进入闭环的门槛。保持readout为基线，没有在确认后改模型。数据含privileged接触候选和源分布限制，不能声称纯图像在线方法。

本轮补入最终E1的10个源文件，逐一匹配原冻结hash；仍有旧机器绝对路径及外部依赖，属于原始归档，并非已移植的跨机器运行包。

证据：[最终E1报告](../experiments/historical_results/e1/mixed_ranker_block_v1/MIXED_RANKER_BLOCK_REPORT.md)、[迁移清单](../experiments/historical_results/MIGRATION_MANIFEST.json)。

### 4. E4：Navigation原生基线、标签和轻量评分器

将队友notebook函数按AST提取为无渲染实验入口，保留原生VMAS物理；原对照18/18缓存结果核对匹配（延迟不要求匹配）。新增安全诊断覆盖初态与原生子步，避免只在环境步末看间距而漏掉中途违规。

建立几何分组数据、真实候选代价与安全标签、模型冻结与确认访问记录、原生回放、独立NumPy重算和逐场景结果表。训练5059参数直接状态/动作评分器，输入为当前状态、目标与候选动作，不输入真实未来，也不加载Navigation LeWM。

后续通过新场景诊断、相对坐标/双机交换一致性、等待/制动候选和代价对照，逐步区分安全筛选、目标排序及候选覆盖问题。离线安全改善没有直接当作闭环成功。

### 5. 闭环验证、目标代价修复和最新压力边界

| 实验块 | 关键结果 | 解释边界 |
|---|---|---|
| 最初60集闭环对照 | 修复评分器0/12到达；真实仿真参考12/12安全到达 | 离线安全排序不够完成导航 |
| 新开发轨迹诊断 | 24轨迹/126状态/9072候选；所有采样状态有安全推进候选 | 仅为有限H8状态诊断，不是完整可行性证明 |
| 48集新确认 | 解析目标代价+冻结风险12/12安全到达；训练ridge0/12 | 预设ridge门槛失败，解析对照另列 |
| 72集四条件鲁棒性 | 解析方案24/24安全到达；原评分器0/24 | 仅6个基础几何、四种相关变体；不是24个独立场景 |
| 本轮36集新压力测试 | 解析方案近距离6/6安全到达；高惯性到达6/6但安全到达0/6；真仿真两条件均6/6安全到达 | **已暴露风险过滤的速度/惯性泛化边界，不能声称方法普遍安全** |

目前最有用的正结果是**已知自由动力学目标代价+冻结学习风险过滤**这一混合基线。它知道仿真参数、忽略接触力，不是学习得到的Navigation世界模型。此前离线解析选择也存在11.90%不安全率；最新高惯性失败进一步限制安全结论。

最新压力测试初态均安全，模型/阈值/候选/物理参数不变。有限候选池真仿真参考能安全完成，不等于全局最优；初始安全候选数量与失败诊断见报告。全部已消耗确认/压力几何及其变体不再用于后续调参。

证据：[目标代价报告](../experiments/e4_navigation/navigation_cost_results/v1/REPORT.md)、[四条件鲁棒性](../experiments/e4_navigation/robustness_results/v1/REPORT.md)、[新压力报告](../experiments/e4_navigation/pressure_results/v1/REPORT.md)。

### 6. 工程复现与协作交付

- 提供便携风险模型JSON、共享runner及逐集比较工具，队友无需访问本机作者权重路径即可运行当前E4基线。
- 本轮从GitHub新clone已发布提交`d2d0af7`，在**同一机器、既有Python环境**完整重跑72集。逐集指标零差异，720项数组字段逐元素一致；这不替代队友跨机器独立复现，也不是新的科学确认集。
- 阶段结果、协议、逐集表、模型/数据hash、失败门槛、环境版本与测试持续上传个人分支。选定23份早期E0/E1文件已迁移，不冒充全量历史包。
- 同步`main`的上游readout提交`a5c2d76`到个人分支，为PR评审准备；原作者贡献保留。没有向main合并，也没有代表队友完成评审或发送消息。
- 建立外部资产清单：3,723个科研文件，重新校验的小文件、沿用声明hash的大文件和未hash文件分开标记。大型原始数据与作者模型没有随Git上传；无共享地址的项明确标为待上传。

## 三、目前能支持的结论

1. 固定、可审计的候选与物理/成功口径比单独看离线评分指标更可靠；初始到位、接触历史与子步间距会影响结论。
2. 现有直接评分器的目标代价排序影响闭环目标推进；解析对照能改善若干新场景，但训练线性修正器尚未提供闭环收益。
3. 学习风险筛选并未充分覆盖高惯性状态；成功样本不能掩盖已观察到的违规。
4. 未获得Navigation LeWM之前，不能评价其预测/规划能力，更不能宣称已验证三机泛化或论文核心模型贡献。

## 四、交付清单

| 材料 | 位置 |
|---|---|
| 个人分支与协作入口 | [COLLABORATION_START_HERE.md](../COLLABORATION_START_HERE.md) |
| 本说明 | `docs/WORK_SUMMARY_2026-10-25.md` |
| PR评审草案 | [PR_REVIEW_READY.md](PR_REVIEW_READY.md) |
| 便携复现说明与模型 | [REPRODUCE_SHARED.md](../experiments/e4_navigation/REPRODUCE_SHARED.md) |
| 同机干净clone证据 | [team_readiness_results](../experiments/e4_navigation/team_readiness_results/v1/clean_clone_verification.json) |
| 大文件索引与共享待办 | [ASSET_SHARING.md](ASSET_SHARING.md) |
| 历史结果与源码 | [historical_results](../experiments/historical_results/README.md) |

原始大数组、训练缓存、作者模型、部分旧脚本和虚拟环境仍保存在本地；一次clone不能重跑全部历史研究。所有指标须回溯相应协议和分母，不能跨任务合并成功率。

## 五、下一阶段任务

1. **队友独立复现和评审：**另一台机器/独立环境完成smoke与全套复现，核对指标与贡献归属，再决定PR合并。
2. **风险边界修复：**基于最新高惯性失败设计新的训练/验证分布与校准方案；保留解析/真仿真基线，修复冻结后再激活全新确认几何。本轮不在压力集上直接调阈值。
3. **模型负责人材料交付：**Navigation checkpoint、加载代码、归一化/时序/动作/agent约定、state head/readout和数据划分；材料齐全后再接LeWM rollout对照。
4. **大文件共享与旧实验移植：**团队确定数据存储与访问方式后，按run提供SHA256及split/consumption记录；移植旧路径时保留原归档，不能覆盖冻结源文件。

**当前状态：**可继续推进直接评分/风险校准、数据与工程协作；Navigation LeWM接入受材料缺失限制。没有把材料等待或未完成事项写成已完成结果。
