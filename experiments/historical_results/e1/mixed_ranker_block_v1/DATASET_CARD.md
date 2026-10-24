# Mixed ranker v1 数据与加载边界

train24场景，内部fit18/selection6；confirmation8场景在模型冻结后才采集。本轮confirmation已消耗，不重新调参。

每组64候选固定顺序：0–15 random_step、16–31 random_block、32–47 readout_CEM、48–63 privileged_contact_block。所有方法使用同一池。按scene分区，禁止按候选拆train/test。

inputs.npz中features[64,21]是唯一小模型特征，predicted_cost[64]为基线；其余规划数组为预测latent、goal latent、probe pose、动作和像素。proposal_id只供分组审计，不是模型输入。

labels.npz保存真实未来state/contact、true_cost、true_T_cost和real_encoded_*等监督，只能用于训练目标或评估；不得进入在线特征。真实初态、goal、direction也是监督/privileged诊断键。

cem_intermediate_UNLABELED.npz为前两轮CEM预测，共2048条，没有真实标签，不计入4096标签。source.npz仅用于仿真复现。

可移植ridge评分：readout_cost/100 + clip(((features-mean)/std) @ beta, -0.25, 0.25)。features必须遵循phase22_common.low_features及feature_names，不能用Navigation输入替代。正式checkpoint须配套protocol/hash/冻结probe；候选模型未通过闭环启用门槛，不能宣称优于readout的已验证闭环模型。
