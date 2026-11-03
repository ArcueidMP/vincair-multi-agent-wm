# JEPA Phase B 六配置选择结果

2026-11-03：full、no_cross、flat的G1–G6共18次配置选择训练及完整验证全部完成，队列正常退出。三个变体均选中 **G6：history=4、60,000步**。这是seed0的配置选择结果；正式18个新seed模型尚未开始。


| 配置 | history | steps | full (cm) | no_cross (cm) | flat (cm) |
|---|---:|---:|---:|---:|---:|
| G1 | 1 | 23,000 | 9.206615 | 8.726264 | 10.510912 |
| G2 | 1 | 40,000 | 7.923732 | 7.605333 | 9.438788 |
| G3 | 4 | 23,000 | 7.166845 | 6.262629 | 8.303133 |
| G4 | 4 | 40,000 | 5.918571 | 5.220350 | 7.177821 |
| G5 | 1 | 60,000 | 6.996445 | 6.955688 | 8.732299 |
| G6 | 4 | 60,000 | 5.069164 | 4.671645 | 6.559518 |

验证使用explore_v2_val全部100局（299000–299099），四延迟0/50/100/200 ms，每延迟7,429决策，15步、stride1，只评最终checkpoint。selection_score以m保存，此处换算cm。全部候选只用explore_v2的2,000训练局、选择seed0，从随机初始化开始。

已核对每个候选的最终步数、完整断点中优化器/调度器步数、最终权重与断点模型一致性、源码及数据manifest身份、checkpoint/result SHA256绑定和完整验证覆盖。原始JSON、日志、权重及完整断点保留在data/，表格与哈希进入文档。数据校验记录：训练match2000/differ0/unknown0，验证match100/differ0/unknown0；本次重新核对manifest与训练元数据绑定。

## 配方与选择规则

维护者[评论6555210796](https://github.com/ArcueidMP/vincair-multi-agent-wm/pull/3#issuecomment-6555210796)批准共同新增G5/G6，并将60,000步设为最终上限。各变体按六配置完整验证原始分数独立选优，完全同分按G1→G6编号；不使用dev/test或中间checkpoint选型，不再扩展步数。G1–G4核验后复用，G5/G6使用独立输出名和完整60,000步OneCycle计划，没有从40,000步断点延长训练。

固定配方：batch128、lr0.001、dim128、latent64、encoder2、predictor4、heads4、head_weight20、SIGReg0.01、projections1024、rollout15、stats_samples5000、AdamW/OneCycle。full/no_cross H4总参数1,289,225，flat H4为1,372,047。训练器、骨干、损失、归一化及冻结Arena文件保持不变。

## 分延迟载荷误差（cm）

### full

| 配置 | 0 ms | 50 ms | 100 ms | 200 ms |
|---|---:|---:|---:|---:|
| G1 | 8.907943 | 9.089280 | 9.258855 | 9.570381 |
| G2 | 7.681412 | 7.816548 | 7.967038 | 8.229928 |
| G3 | 6.891867 | 7.034317 | 7.214598 | 7.526596 |
| G4 | 5.608173 | 5.745389 | 6.000727 | 6.319994 |
| G5 | 6.745404 | 6.862633 | 7.072005 | 7.305739 |
| G6 | 4.703157 | 4.882942 | 5.159875 | 5.530682 |

### no_cross

| 配置 | 0 ms | 50 ms | 100 ms | 200 ms |
|---|---:|---:|---:|---:|
| G1 | 8.470568 | 8.634979 | 8.762362 | 9.037146 |
| G2 | 7.365561 | 7.495046 | 7.656097 | 7.904629 |
| G3 | 5.942753 | 6.108305 | 6.330498 | 6.668962 |
| G4 | 4.855040 | 5.031869 | 5.292806 | 5.701686 |
| G5 | 6.699826 | 6.815394 | 7.024646 | 7.282884 |
| G6 | 4.290493 | 4.444161 | 4.733994 | 5.217932 |

### flat

| 配置 | 0 ms | 50 ms | 100 ms | 200 ms |
|---|---:|---:|---:|---:|
| G1 | 10.276874 | 10.423960 | 10.536676 | 10.806136 |
| G2 | 9.221022 | 9.350719 | 9.471089 | 9.712322 |
| G3 | 8.041157 | 8.182902 | 8.348752 | 8.639719 |
| G4 | 6.886310 | 7.063942 | 7.235318 | 7.525714 |
| G5 | 8.493780 | 8.643501 | 8.769954 | 9.021961 |
| G6 | 6.259893 | 6.436291 | 6.620415 | 6.921472 |

## 计算开销

| 变体 | G1分钟 | G2分钟 | G3分钟 | G4分钟 | G5分钟 | G6分钟 |
|---|---:|---:|---:|---:|---:|---:|
| full | 77.48 | 132.10 | 89.10 | 154.85 | 211.32 | 225.03 |
| no_cross | 75.66 | 136.52 | 82.31 | 146.85 | 189.86 | 226.01 |
| flat | 49.17 | 77.40 | 50.82 | 92.10 | 119.01 | 137.00 |

新增G5/G6共6次训练实测 **18.4707 GPU小时**，接近原约19小时估计；全部18次选择训练累计37.8768 GPU小时。以上均不包含验证、数据加载与统计时间。

## W-ar交接与后续正式训练

full最终history为 **4**，骨干继续使用full mask、dim128、latent64、encoder2、predictor4、heads4、dropout0；W-ar CLI及backbone-options均传history4。结构与factory详见[PHASE_B_FULL.md](PHASE_B_FULL.md)。W-ar自己的batch、lr和steps沿用其获批网格，不照搬JEPA损失或60,000步选择。

后续JEPA正式训练固定各自G6配置：三个变体×新seed1/2/3×500/2,000局，共18个模型；两种数据预算使用相同history4和60,000步，不复用选择seed0权重。正式test和L1/L2/L3交由维护者在同一台机器统一评测。当前尚未启动正式训练，也未运行新的dev/test。

当前no_cross的验证分数低于full，是单个选择seed上的open-loop结果，不据此推断正式多seed、L2或任务成功率J的排序。

## 文件与身份

- 网格及汇总：`data/uav_arena/results/jepa/phase_b_grid.json`、`phase_b_{full,no_cross,flat}_selection.json`。
- 候选权重/断点：`data/uav_arena/participants/jepa/phase_b_select_{variant}_g{1..6}_s0.pt`和`.resume.pt`。
- 官方验证：`data/uav_arena/results/phase_b_select_{variant}_g{1..6}_s0_openloop.json`。
- 队列收据：`data/uav_arena/results/jepa/phase_b_select_{variant}_g{1..6}_s0_queue.json`。
- train/val及runner日志：`data/uav_arena/logs/`。
- 迁移前99文件归档：`data/uav_arena/results/jepa/archive/phase_b_g1_g4_2026-11-02/manifest.json`，逐文件SHA256已核对。

数据manifest SHA256：

- explore_v2：`cb196ded5be80b12039fc65e1c3797ab50113394046a7f03b8c1de97ac7959c0`
- explore_v2_val：`2e6a5c8fd32fcbff297110248570e78c7f4ec6dd557be33fa383555312e8cab0`

所有结果使用同一训练源码身份；以下文件对应归档和完整断点内记录：
- `train.py`：`690ebca4629ce54c307c1aa5ca70b6a1035dce932d3be22788938378c0dd7130`
- `objective.py`：`5fdee25ff720a3fb276b22e05bd76609060e75d2cff11a1f570ff548097e61b0`
- `model.py`：`56f99a3ff1cd826e1bb51222bcb2f3850a1461846e171b98ce3f00430505c807`
- `backbone.py`：`0a03e35980ecd9ab361d1aa42298f9906b0b9720a21659f722cc925be8a73bf0`
- `training_checkpoint.py`：`9b8af9a3bba2340a9040596f1275a615fad7ed8de8e078474fb276fbc83786f9`

### full：checkpoint与结果SHA256

- G1 checkpoint：`c1dbee9b4438f240c6109d42c21e21e1a7bc2e380646f151ad95fe4a5a93d2cc`；result：`b87dc7d3c1b2a98a8462fd4c8d7b6d003ca6aadc9d21be28fa2367558d873e89`。
- G2 checkpoint：`237823eebf70b79e21515617d7adbc028daa286072af867e9fc2752a61222f2d`；result：`212f49050d98f61ec90be734b626611afd56a6e2ad68001c35fcd8bd3ed78986`。
- G3 checkpoint：`d35ce6793d55654c36555ec68ebf05619b70bb1ad3161a2652c930ada695abd0`；result：`b9474d167da25bfb4fbdf9287d247401319c9f08df841ea4dd64d30ebad88aff`。
- G4 checkpoint：`cb84b1ff80f2a7f18b7417e7738b98bb07be9364317cffb3c9edce82277501dc`；result：`cad3e4f4375c310b9334598bbfab1e4ea8168a2f1e42392054bb608cd8fd4915`。
- G5 checkpoint：`65661cbd173d653a7ab65954a7ebc9ba33b66ad1d6a958c378d735647c8db8f9`；result：`e49ce0a60a59b89ff7e3aff48318be6f76e0a6a5d680e822560bd7075fec809e`。
- G6 checkpoint：`acc5a91354f8fa47d52cfffb64be8371e81acd41c9212d43cc3da66f7a378524`；result：`11cbb8a4a5d667e3dac30d10fce3a2e5cd1132a0028e77dbff9bba539b19b099`。

### no_cross：checkpoint与结果SHA256

- G1 checkpoint：`9028240c2b806d8616fc9b4c6bba6ec935b7a4fb536868f7b2756336bed4fa0e`；result：`53a481b8d0115d3f045804ac54dbd7bef7e1fe1e7b0d89f98d2bcdd9e203c0df`。
- G2 checkpoint：`327316c01a116e1909e0e49ce8fcafc6fab589257b44bcbad682f063d1f53270`；result：`6bd33dce5b08c346a649e15547b1833abd16beee828b7dedb5d0e91ced415e24`。
- G3 checkpoint：`3caa0fe580c9e05a67f9ae8a9da1263c9d50a5487aa778bbc18159974d67012c`；result：`e74347827541bb211034f9452d2c283bc34fd37d3d2d852d03ee914000076668`。
- G4 checkpoint：`a3f3dc3beae1339b21c36ff7527db85063453c69aa02035832b85ec98eefc567`；result：`9d225d8f319a4bd9d7cd689b96ac1791434b58987ea01374101064fcfe256e3f`。
- G5 checkpoint：`77ca8884690b356c71b2fba29c2dcdf80475109a0cf2169315e2bd4955b43630`；result：`249164eda33e3b256392b92e3d32bf5b4f0567c94a5afc0a27526f4b64348976`。
- G6 checkpoint：`cb77dc4c8f9dc488bf4692ab5e18d8586c8286a2e5bf821569918379b0887755`；result：`633844b7d1f823d0ca590c2367ea0e40fc3289b413f2fafbdeed02e2a44aedfe`。

### flat：checkpoint与结果SHA256

- G1 checkpoint：`01a499581ec06278c584e2931e0273a78d3e817acb594c60e4c6ab3e034461cf`；result：`e77823861a632600044f263d970a4846a955d6ab148876ea7796a84afdaa7fa3`。
- G2 checkpoint：`3fd988b2e6c72f6bd377fd7cabc0b588e2fdae4bd27f850afb900066e40d3f6d`；result：`07960e15b0d6e6377c570d0fa7c75667e93f3dde80dee3490d7823c69cb654c0`。
- G3 checkpoint：`82b1c3ef8eea55ac022bccd3daf1aee7f0ce58688f24b50bc338a0427e111467`；result：`0e87839f432e3afc1d07e642981562bc2df1c67786c0824a2f0e82224cf5d7a6`。
- G4 checkpoint：`6f8d35e98e4c165afa0c45be94dae7752dc14d15b1fa23e44c826ef0ffb26557`；result：`881bce5ac7e6ac65a74a8c73b44d90f9e97cff387d0d995d6350aa2fd12c9c36`。
- G5 checkpoint：`3df5cad76a03950dbd9eb7da2323733cae4c62b25736d9bf3334163b611ee33b`；result：`98bcf376744de35606aa7020dc013e51cc3bb26f65c821e77e42e58c9a2769f4`。
- G6 checkpoint：`f4053b56c88742ecff05c742b046221c484ad5ba9feb22a7d4aeb486fc1fddb7`；result：`801382b2e703e448cf8db9b01c43d7354bc5b97e76b4089a026888c42bd8efb4`。

## 本次PR检查

CPU定向测试：骨干、参赛接口、选择队列和完整断点共75通过。六配置队列新增审批/硬上限检查、默认只读、不兼容40k断点拒绝、六候选原始分数选优与同分编号顺序测试。git diff --check通过；本次没有重新训练或评分。
