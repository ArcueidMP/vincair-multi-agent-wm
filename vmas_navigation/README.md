# 本地数值 Navigation JEPA pilot

此模块将Navigation每机数值观测适配为agent tokens，复用根模块的SIGReg实现；不修改`vmas_pusht`，不加载PushT/image/作者Navigation权重。

**状态：工程训练闭环已通过，500步模型位置预测资格未通过。不能直接接planner。** 队友当前采集/架构配置尚未提供，因此协议标为`LOCAL_NAV_NUMERIC_PILOT_V1`。

## 运行

在仓库根目录、原有CUDA项目环境执行（需要已有torch/numpy/vmas/einops等依赖，无新增生产依赖）：

```bash
python -m vmas_navigation.pilot --output /tmp/nav-jepa-pilot-01
python -m vmas_navigation.verify --run /tmp/nav-jepa-pilot-01
python -m vmas_navigation.report --run /tmp/nav-jepa-pilot-01
```

输出目录必须不存在，防止覆盖旧实验。会真实采集64集CPU轨迹、训练500步CUDA模型、验证并保存权重/优化器/RNG/预测。当前是固定短程runner，没有正式长训练resume CLI；不能将权重文件存在等同于具备断点恢复入口或合格规划能力。

## 输入输出

- `state [E,T+1,N,4]`：每机自身可观测pos/vel，从原生18维obs前4维提取；集中式集合输入。
- `action [E,T,N,2]`：物理动作，分量[-1,1]，frameskip1，dt0.1。
- `native_observation [E,T+1,N,18]`：完整原生观测，仅保存供后续协议对齐；goals另存元数据。
- `substep_state [E,2T,N,4]`：真实原生子步；不当作训练输入。
- history3，训练自主rollout3，验证rollout8。Head输出归一化位置/速度，评估时用训练集统计恢复物理单位。
- 时间embedding沿frame共享；没有agent身份/固定实体位置embedding。支持动态N的API形状不代表三机泛化，当前仅采集/训练双机。

## 结果与对齐

见[本轮报告](../experiments/e4_navigation/navigation_training_results/v1/REPORT.md)与[训练补充计划](../docs/NAVIGATION_TRAINING_PLAN.md)。公开代码/小结果可重跑，原始dataset.pt/checkpoint/预测NPZ仍在本地并由artifact hash清单标识；本次没有在Git中上传二进制权重。

待组员给出collector/config/architecture commit后，核对观测语义、策略混合、数据规模、history/frameskip、网络/loss/normalization、agent数和head训练。旧PushT的T字段、15D state、模型步frameskip5等不能默认为Navigation协议。
