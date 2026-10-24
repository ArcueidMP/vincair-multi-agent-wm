# 便携 Navigation 基线复现

这是双机 VMAS Navigation 的**直接风险评分器+解析目标代价**基线，不需要 Navigation LeWM checkpoint。源码来自本分支 `experiments/e4_navigation/`，便携模型为 `shared_models/invariant_safety_v1.json`。

## 运行环境

原实验在 WSL2 Ubuntu、RTX4060 Laptop8GB 上运行。准确 Python/Torch/NumPy/VMAS 版本记录在 [tested_environment.json](robustness_results/v1/tested_environment.json)。当前 runner 使用 CUDA，不支持直接在没有 CUDA 的机器上执行全套。优先复用团队已有环境；不要为了复现更改现有实验环境或依赖源码。

运行前检查：

```bash
python -c "import torch,numpy,vmas; print(torch.__version__, numpy.__version__); print('CUDA:',torch.cuda.is_available())"
```

Python 模块 `numpy`、`torch`、`vmas` 是该基线的主要依赖。它通过 AST 加载已保存在仓库的 notebook 函数，无需启动 Jupyter 或执行 notebook 安装单元格。其他研究模块/完整测试可能需要项目原有依赖；当前工作没有新增生产依赖。

## 第一步：独立便携 smoke

在仓库根目录执行，输出目录必须不存在：

```bash
python experiments/e4_navigation/shared_navigation_model.py \
  --output /tmp/lewm-portable-smoke-01
```

运行三种方法各5步，并为每条轨迹做新的原生状态/子步回放。查看 `results.json` 与各子目录 `result.json` 中的 replay error。Smoke只验证接入，不是导航成功率评估。已核查 JSON 模型参数与原 checkpoint 精确一致，三个 smoke 的状态/动作/候选/预测与原模型逐元素一致。

## 第二步：复现冻结的72集鲁棒性实验

```bash
python experiments/e4_navigation/shared_navigation_model.py \
  --protocol experiments/e4_navigation/robustness_results/v1/protocol.json \
  --full --output /tmp/lewm-robustness-repeat-01
```

6个基础几何×4个相关条件×3种方法，共72集、每集最多200步、H8/72候选，输出可能较大。保持同样 VMAS 版本、CUDA设备、模型和协议。对照仓库 [episodes.csv](robustness_results/v1/episodes.csv) 的 geometry/arm/seed、步数、成功/安全标志、最小间距；耗时和延迟不要求精确一致。完整新机运行尚需队友独立复核，不用本机 smoke 代替跨机器验证。

如输出有差异，保留日志/轨迹和版本，先核对模型 SHA256、原生物理设置、动作/goal定义；不要通过改阈值或删失败场景让结果看似匹配。

## 模型来源与数据

- JSON仅含我们在 `safety_repair_v1` 训练的5059参数评分器、归一化统计、冻结策略及源文件hash；epoch63，risk threshold0.5、minimum margin0。
- `export_shared_model.py` 从原小模型导出，`shared_navigation_model.py` 只做严格形状加载和推理，不下载外部权重、不训练。
- 原始作者 LeWM/PushT checkpoint 与大规模数据没有嵌入 JSON。
- 训练源报告：[safety_repair_results](safety_repair_results/v1/REPORT.md)；目标代价对照：[navigation_cost_results](navigation_cost_results/v1/REPORT.md)；最新鲁棒性：[robustness_results](robustness_results/v1/REPORT.md)。
- 全套历史实验的原始 NPZ、作者模型与部分旧项目路径仍需共享数据包；当前便携入口只解决这套 E4 混合基线的复现。
