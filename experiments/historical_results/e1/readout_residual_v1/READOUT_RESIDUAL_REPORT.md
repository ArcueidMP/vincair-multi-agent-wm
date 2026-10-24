# E1 小型 readout 残差评分实验

## 结论

本轮没有取得优于 readout 的结果。三个残差模型均已实际拟合，但没有改变训练内部8场景的首选候选；按照结果产生前固定的选择规则，保留 **readout 零修正**。最终 residual_ranker.npz 的 beta 全零，是基线兼容产物，不能当作新的有效学习模型。

## 实施范围

- 冻结 world model 与官方 probe，直接使用已有预测 latent，不重新运行 world model 或物理仿真。
- 在旧train40场景中固定前32场景拟合、后8场景内部选择；无场景交叉。三个 ridge 正则系数为0.1、10、100，包含零修正基线；并列首选基线。
- 9维输入：预测位置相对goal差4维、角差sin/cos、readout cost、联合位置误差norm、角差绝对值。没有真实未来状态、接触标签或真实代价进入特征。
- 场景/组/候选等权的组内中心化残差ridge，训练标准化仅由拟合集计算；修正幅度预先限定为±25真实代价单位。
- protocol先写入，selection_frozen写入并保存checkpoint后才首次打开本轮validation候选NPZ。零修正被选中，所以没有再拟合非零的40场景最终模型。
- 训练和验证共297组、4752候选；旧test候选NPZ及test_evaluation.npz均未打开。旧validation早已用于开发，本轮不能称独立确认。

## 训练内部选择结果

| 方法 | 场景平均 regret | 场景平均 PRA | 相对readout首选改变组数 | 平均绝对修正（真实代价单位） |
|---|---:|---:|---:|---:|
| readout | 4.0458 | 85.66% | 0 | 0 |
| ridge 0.1 | 4.0458 | 85.63% | 0 | 7.2332 |
| ridge 10 | 4.0458 | 85.63% | 0 | 0.1068 |
| ridge 100 | 4.0458 | 85.66% | 0 | 0.0107 |

三个拟合结果的beta范数均非零，线性正规方程残差<3e-18；并非训练没有执行。较弱正则0.1的平均绝对修正7.23、最大17.22，但各组首选仍未变化。PRA、pairwise softplus和scaled MSE均已保存；本轮按预定top1 regret选择，不在看到PRA后更换选择目标。

## 冻结后的验证

validation8场景/50组：原readout与选中的零修正均为 **regret3.9836、PRA85.94%**。两者得分逐元素完全相同，这只是基线复现，不能作为残差学习有效的证据。三个未选残差模型没有在validation上逐个比较，避免把该集继续用于挑配置。

## 验证与产出

独立从原821维实现恢复pose特征，在train上重新拟合，重现所有内部选择regret/PRA；核验场景分区、正规方程、得分有限值、零修正等价、checkpoint重载、probe hash不变，全部通过。未执行闭环、没有新成功率或新测试集结论。拟合与一次验证主程序用时5.05秒，CPU完成；无需GPU训练或新增权重。

输出：protocol.json、selection_frozen.json、residual_ranker.npz、inner_models_audited.npz、summary.json、verification.json、access_log.json、train_scores.npz、validation_scores.npz。原始权重及首版ranker保持不变。源代码归档 scripts/phase20_v1。

## 下一步

按预定的失败分支，保留 readout，优先做 **MultiPushT 候选分布审计与定向补采**：固定训练新场景，在相同状态下比较原逐步随机候选与CEM/contact/block proposal，保存完整真实重放标签、预测latent与readout误差，先确认缺口再决定是否重训。候选生成如使用真实接触几何，必须标记为privileged诊断，不能混同在线可用方法。将新确认场景独立封存，不能因本轮失败继续对旧test调参。

本轮没有证据支持扩大网络、延长训练或启动大规模闭环。Navigation E4仍等待组员仓库及模型/环境契约，本任务无需新材料。
