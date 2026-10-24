# 原始数据与模型共享清单

本轮只生成可审计索引，没有创建新的 HF 数据仓库或上传原始权重/大数组。

## 已共享

- Git中包含源码、逐集/候选汇总、协议、验证与消费标记，以及我们自行训练的5059参数直接风险模型JSON。
- 当前E4解析/风险混合基线的便携入口可不依赖作者checkpoint运行；本机干净clone的完整72集复现已通过。
- 历史E0/E1只迁移了选定报告和最终E1源码，不是完整历史原始数据包。

## 本地资产索引

见[ASSET_INDEX.json](../experiments/e4_navigation/team_readiness_results/v1/ASSET_INDEX.json)和[files.csv](../experiments/e4_navigation/team_readiness_results/v1/assets_files.csv)。路径相对于实验资产根目录，不包含账户缓存或私聊图片。

索引快照覆盖`results/e0`、`results/e1`、`results/e4`内指定科研文件：3,723个文件、345,669,154字节；3,602个小文件重新计算SHA256，96个大文件沿用已存在manifest中的声明hash，25个未计算hash。**声明hash不等于本轮重新校验。** 另对原阶段manifest中的6份模型/配置文件和两个MultiPushT readout进行了本地hash核验。

索引在新压力实验结束前生成，压力结果的原始轨迹另由其[trajectory_sha256.json](../experiments/e4_navigation/pressure_results/v1/trajectory_sha256.json)索引。此快照也不覆盖`data/datasets`全部训练资产或缓存，不能当作整台电脑数据清单。

`author_weight_inventory`按原下载manifest固定的HF revision构造来源链接，并明确标注PushT任务；本轮没有重新联网下载或声称这些是Navigation权重。单智能体`r-lewm-uav`不包含在当前Navigation模型清单中。

## 待团队完成的共享动作

1. 选定有访问控制和容量约定的HF数据仓库或团队数据盘；目前`shared_download_url=null`表示尚无共享下载入口。
2. 按独立实验run打包原始inputs/labels/predictions/metadata/protocol与split/consumption记录；保留文件SHA256，禁止重分割已消费确认集。
3. 先共享最小复现实验所需文件，再决定是否迁移全量缓存。作者模型优先引用原始发布源；重新分发前核对来源与许可，不能把项目缓存一并上传。
4. 接收方校验下载文件SHA256，记录环境和复现结果。不要仅凭文件名或大小确认模型匹配。

Navigation专用checkpoint及其加载代码、配置/归一化、history/frameskip、动作/agent约定、state head/readout和数据划分仍未提供；此项由模型负责人交付。当前PushT权重不能补位。
