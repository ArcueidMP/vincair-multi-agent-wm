"""Publish feasibility and failed model qualification without planner claims."""
import argparse,csv,json
from pathlib import Path
from vmas_navigation.pilot import write,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    s=json.loads((root/'summary.json').read_text());v=json.loads((root/'verification.json').read_text());d=json.loads((root/'native_safety_diagnostics.json').read_text());rows=[]
    for name,m in s['validation']['metrics'].items():
        for k in range(8):rows.append(dict(method=name,model_steps=k+1,physical_seconds=(k+1)*.1,position_L2=m['position_L2_by_k'][k],velocity_L2=m['velocity_L2_by_k'][k]))
    with (root/'prediction_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    table='\n'.join(f"| {name} | {m['position_L2_by_k'][0]:.5f} | {m['position_L2_by_k'][7]:.5f} | {m['velocity_L2_by_k'][7]:.5f} |" for name,m in s['validation']['metrics'].items())
    c=s['collection'];metrics=s['validation'];report=f'''# 本地 Navigation Entity-Token JEPA 最小训练闭环

## 结论

**工程闭环通过，模型预测资格未通过。** 已真实完成64集CPU采集、500步GPU训练、保存/重载与独立验证；这是数值观测Navigation模型的本地原型，不是队友当前新架构的严格复现，也不是已合格的LeWM规划器。没有进行闭环规划或三机效果实验，没有为了通过门槛临时增加训练步数。

| 位置/速度预测方法 | 1步位置L2 | 8步位置L2 | 8步速度L2 |
|---|---:|---:|---:|
{table}

单位为VMAS物理单位；1步0.1秒、8步0.8秒。80个验证窗口来自4个几何、16集轨迹，每集5个预先固定起点；窗口相关，不能作为80个独立场景。最后500步checkpoint按固定预算保存，没有按验证结果挑选最优模型。

JEPA位置误差高于persistence，不能仅凭训练损失从前50步均值{s['first50_loss_mean']:.3f}降至后50步{s['last50_loss_mean']:.3f}接入规划。编码状态头位置误差1步{s['validation']['metrics']['encoded_readout']['position_L2_by_k'][0]:.3f}/8步{s['validation']['metrics']['encoded_readout']['position_L2_by_k'][7]:.3f}；这提示应先检查重建和训练充分性，尚不能唯一归因于某个网络部件。latent均值标准差{metrics['encoded_latent_mean_std']:.4f}只是诊断，不是完整无坍塌保证。

## 协议与适配

- 协议ID:`LOCAL_NAV_NUMERIC_PILOT_V1`。队友尚未提供当前collector/config/architecture commit，不能声称采集协议完全一致。原始`Multi_Agent_LeWM.md`保留，不覆盖；执行调整见训练补充计划。
- 输入为每机原生18维observation中的前4维`[x,y,vx,vy]`，以集中式集合输入；完整18维及静态goals另存。没有把T物体字段、像素权重或旧Navigation非entity模型直接塞进来，也不把集中式输入叫分散策略。
- 每机共享embedding/action/head，只有时间embedding，没有agent身份/固定实体位置embedding；历史3帧，frameskip1，训练自主rollout3步，验证8步。无T token，goals不进入动力学输入。
- dim64/latent32、encoder1层/predictor2层、SIGReg64投影、{s['parameters']}参数、AdamW lr3e-4、batch128、seed0、固定500步。小配置不同于现有PushT模块dim128/SIGReg1024，不等于组员正式配置。
- Head同时在encoded和predicted latent上训练，并有delta监督。损失与权重完整记录于protocol。模型从头初始化，不宣称image-to-state预训练权重迁移成功。

## 数据与泄漏控制

16个NEW几何(seed2026105301)、每个4类采集策略，共64集x128步=8192 transitions。12几何/48集训练，4几何/16集验证。训练/验证整几何分离；训练统计按4维跨agent共享，没有用验证集估计归一化。验证可作后续开发诊断，不是未使用的测试集。

采集策略:noisy PD、priority PD、yield PD、random OU。CPU四环境并行，原生VMAS1.5.2 dt0.1/substeps2/radius0.1/mass1/drag0.25，动作分量[-1,1]；到达后仍采满长度，不自动重置。所有原始状态、动作、观测、原生子步与episode manifest保留。没有读取此前已消耗压力/确认轨迹作为训练样本；未知作者预训练集的交集没有声称已排除。

## 本机资源实测

| 项目 | 实测 |
|---|---:|
| CPU采集+第一次独立回放墙钟 | {c['wall_seconds']:.2f}s |
| 同段进程CPU time | {c['cpu_process_seconds']:.2f}s |
| 500步GPU训练（含验证） | {s['training_wall_seconds']:.2f}s |
| 本轮PyTorch分配器峰值（含验证/重载） | {s['peak_cuda_allocated_MiB']:.2f}MiB |
| 脚本内部端到端墙钟 | {s['wall_seconds']:.2f}s |

脚本端到端不含Python导入、工具启动和额外独立审计。分配器统计不含CUDA context/桌面/其他进程，不是整卡总占用。队友所说30–45分钟CPU采集、3–4小时laptop GPU训练属于不同且未给定的规模/配置，不能用此小pilot作同预算对比或承诺正式训练耗时。

## 独立验证

59项针对性测试通过：时间窗/动作对齐、训练统计、置换一致性、变agent数API形状、有限梯度、原生采集/观测契约与采集后训练seed隔离。三机API测试不等于三机训练或泛化。

首轮v1已保留但被替代：发现VMAS初始化将CUDA RNG置为42，原先在采集前manual_seed(0)不足以确保训练CUDA seed0。v2在采集后重设CPU/CUDA种子，并记录与验证两者均为0；未因数值结果挑种子。v1原始协议/权重/指标和精确源码快照均保留，v2为正式发布的本地pilot结果。

额外从新原生环境重放全部64集，状态/子步误差0；原生18维前4维与保存state逐元素一致。NumPy重算所有物理误差，最大差{v['max_numpy_physical_metric_error']:.3g}。CPU重载只编码初始3帧、不编码真实未来，再自主预测，与GPU保存结果最大差{v['cpu_context_only_vs_saved_gpu_prediction_error']:.3g}。checkpoint含优化器和RNG状态，但当前pilot CLI尚无断点恢复参数；正式长训练前需补周期保存/恢复验证。

模型端点安全诊断false-safe0并不表示可靠：80窗口内真实原生不安全{d['jepa']['true_native_unsafe_windows']}，预测不安全{d['jepa']['predicted_unsafe_windows']}，false-unsafe{d['jepa']['false_unsafe_vs_native']}。端点预测不能证实原生子步全程安全，未用来认证规划。

## 交付与下一步

新增`vmas_navigation/`独立模块、测试、原始数据/训练/预测报告与哈希。源码和小结果提交个人分支，原始dataset/checkpoint/predictions留在本地结果目录，不伪称已向队友下载权重或已共享全部大数组。

下一块优先做固定预算的训练充分性/状态头诊断，验证位置重建和多步误差能否超过简单基线；使用现有训练/验证分组作开发，不把它叫新确认。队友给出采集脚本、观测定义、history/frameskip和新架构commit后再做协议差异表、决定正式数据规模。预测资格通过后才接同物理时长/候选预算的E1排序及E4闭环，最后考虑三机。
'''
    (root/'REPORT.md').write_text(report,encoding='utf-8')
    write(root/'artifact_sha256.json',{p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name not in ['artifact_sha256.json','PUBLICATION.json']})
    print('Report, physical metric CSV and artifact hashes saved.')
if __name__=='__main__':main()
