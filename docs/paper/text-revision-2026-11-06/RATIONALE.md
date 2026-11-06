# 修改依据与审核边界

## 原稿中的问题 → 本次修改

| 问题 | 处理 | 审核位置 |
|---|---|---|
| 引言、贡献、结果反复完整提出同一问题 | 引言建立问题，贡献列交付物，结果直接给证据 | BEFORE_AFTER.md 的 Introduction / Results |
| 定义后仍完整展开跨智能体传播过程 | 定义一次，后文复用 cross-agent error / L2 | Evaluation Layers 与 Results |
| 图例、图注与正文逐项复述 | Fig. 1/5/6 图注聚焦对象、统计及必要类别 | 对照末三项 |
| 结果、机制解释和局限混在同一段 | 结果先行；重规划解释与耦合边界集中到已有 Discussion 段 | JEPA/data-budget findings 与 Discussion |
| 30 WM / 16 physics models 容易被误读成架构数量 | 明确实例组成、候选质量范围与步长 | Data and training / Reference rows |
| 防御性铺垫和重复强调 | 用实验条件与直接观察替代重复辩护 | 全文逐段对照 |

## 技术澄清（来自作者确认）

- 30 = 5 种预测器变体 × 2 个数据预算 × 3 个新种子，不是 30 种架构。
- 16 = 10、12、…、40 g 的候选质量实例，保留最近 0.5 s 的选择窗口。
- 仿真实际调用 Lee controller；其存在与规划器/预测器对内部状态的访问权限分开描述。
- Smith predictor 类比保留，引用由作者后续补充，不虚构来源。
- Table II 已有布局 bootstrap 区间定义，不再重复塞入表头。
- full/no_cross 的局部排序保留，移除将其渲染为推翻总体关联的修辞；PETS/flat 的敏感性限定仍保留。

## 参考的写作原则

阅读的是下列论文的相关正文，借鉴信息组织方式；未搬用其文字、结果或为本研究新增引用。

- [Objective Mismatch in Model-based Reinforcement Learning](https://proceedings.mlr.press/v120/lambert20a.html)：把研究问题落实到具体比较与证据。
- [PETS](https://arxiv.org/html/1805.12114)：稳定术语、区分概念功能、明确实验比较对象。
- [CrazyMARL](https://arxiv.org/html/2509.14126v1)：将条件、对照、观测结果连接，配置由表格集中承载。
- 已有 research-paper-writing / academic-humanizer 的结构与保真原则；参考 [sciwrite-prose](https://github.com/dvdsosa/scientific-writing-skills/tree/master/skills/sciwrite-prose) 的“先提取信息要点、去重，再改句子”方法。

这些是编辑依据，不是论文结果正确性的额外证据；也不意味着原论文的所有写法都应照搬。

## 保持不变

所有 TODO 所在行、表格、公式、引用次序/次数、label、图像输入路径均程序核对一致。Fig. 2 与图注原样保留。图片文件只是为复现预览附入包中，本 PR 不重新设计图片。

已有 TODO 中的未完成科学论述未改写，不能把本 PR 视作最终投稿批准。8 页是当前本地模板与素材下的结果，TODO 完成后需重新编译。
