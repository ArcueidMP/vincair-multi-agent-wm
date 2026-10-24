# 历史 E0/E1 成果迁移

此目录补充当前个人分支中的早期成果，避免团队只看到最新 E4 正结果。

- `e0/`：VMAS PushT E0阶段状态、MultiPushT真仿真15集pilot与readout smoke报告；E0整体不能据此标为全部完成。
- `e1/mixed_ranker_block_v1/`：最终混合候选研究报告、数据卡、冻结协议/来源hash、确认汇总和当时验证记录。24训练+8确认场景、4096候选标签，确认regret改善6.69%，预设闭环门槛失败。
- `e1/mixed_ranker_block_v1/source/`：10个原阶段源文件，与原冻结 `source_sha256.json` **逐一字节匹配**；包含原机器绝对路径与外部依赖，尚未做移植或在新机器重跑。
- 其他E1报告保留readout残差和候选覆盖诊断，避免重复已知失败路线。

`MIGRATION_MANIFEST.json`记录每个文件原始/发布SHA256。报告文本只统一UTF-8/LF；原文件仍保存在本地。迁移不是重新验证全部历史数字；原始数据、全部中间阶段脚本/轨迹、模型、readout及资产不全在此目录，不能把它称为完整历史实验包。

完整早期运行环境基于原项目代码与本地补丁，其物理/成功定义不同于当前原生Navigation实验。禁止混合统计，禁止把PushT checkpoint当作Navigation权重。所有已用确认集仍视为已消耗。
