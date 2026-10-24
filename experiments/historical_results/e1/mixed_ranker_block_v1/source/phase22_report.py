"""Independent confirmation-score audit and final block report (Windows NumPy)."""
from pathlib import Path
import json,hashlib,shutil,ast
import numpy as np
ROOT=Path('D:/资料库/03_开发项目/air traffic/experiments/lewm');OUT=ROOT/'results/e1/mixed_ranker_block_v1'
LOCAL=Path(__file__).resolve().parent.parent/'outputs'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
protocol=read('protocol.json');frozen=read('selection_frozen.json');summary=read('confirmation_summary.json')
trainmeta=read('train_data/metadata.json');confirmmeta=read('confirmation_data/metadata.json');closed=read('closed_loop/summary.json')
for name in ['train_verification.json','confirmation_verification.json','model_verification.json']:assert read(name)['status']=='passed'
assert hashlib.sha256((OUT/'selected_scorer.pt').read_bytes()).hexdigest()==frozen['checkpoint_sha256']==summary['checkpoint_sha256']
with np.load(OUT/'selected_scorer_portable.npz') as z:portable={k:z[k] for k in z.files}
with np.load(OUT/'confirmation_scores.npz') as z:scores={k:z[k] for k in z.files}
assert scores['truth'].shape==(16,64)
manifest=read('confirmation_data/manifest.json');raw=[];numerical_error=0.
for scene in manifest:
    for g in scene['groups']:
        directory=OUT/'confirmation_data'/g['name']
        with np.load(directory/'inputs.npz') as z:
            x=z['features'];base=z['predicted_cost']/100;pred=base+np.clip(((x-portable['mean'])/portable['std'])@portable['beta'],-.25,.25)
        with np.load(directory/'labels.npz') as z:
            truth=z['true_cost'];T=z['true_T_cost'];Tstart=float(z['true_T_cost_start'])
        idx=len(raw);numerical_error=max(numerical_error,float(abs(pred-scores['selected'][idx]).max()))
        assert np.allclose(pred,scores['selected'][idx],rtol=0,atol=1e-10)
        assert np.array_equal(truth,scores['truth'][idx])
        raw.append(dict(scene=scene['scene'],name=g['name'],baseline_regret=float(truth[base.argmin()]-truth.min()),
            selected_regret=float(truth[pred.argmin()]-truth.min()),baseline_T_gain=float(Tstart-T[base.argmin()]),selected_T_gain=float(Tstart-T[pred.argmin()])))
deltas=[]
for scene in range(8):
    group=[q for q in raw if q['scene']==scene]
    a=np.mean([q['selected_regret'] for q in group]);b=np.mean([q['baseline_regret'] for q in group]);deltas.append(float(a-b))
    assert abs(a-summary['results']['selected']['scenes'][scene]['regret'])<1e-10
    assert abs(b-summary['results']['readout']['scenes'][scene]['regret'])<1e-10
assert np.allclose(deltas,summary['regret_delta_by_scene'],rtol=0,atol=1e-10)
better=sum(d<0 for d in deltas);worse=sum(d>0 for d in deltas);ties=8-better-worse
baseline=summary['results']['readout']['mean'];selected=summary['results']['selected']['mean']
rel=(baseline['regret']-selected['regret'])/max(baseline['regret'],1e-8)
gate=frozen['kind']!='zero' and rel>=.1 and better>=5 and selected['T_gain']>=baseline['T_gain']-1
assert gate==summary['closed_loop_gate']==False and closed['status']=='skipped'
trainmanifest=read('train_data/manifest.json');trainseeds={q['seed'] for q in trainmanifest};confirmseeds={q['seed'] for q in manifest}
assert len(trainseeds)==24 and len(confirmseeds)==8 and not trainseeds&confirmseeds
hashes={}
for split in ['train','confirmation']:
    for scene in read(f'{split}_data/manifest.json'):
        for g in scene['groups']:
            for name,key in [('inputs.npz','input_sha256'),('labels.npz','labels_sha256')]:
                p=OUT/f'{split}_data'/g['name']/name;h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==g[key]
                hashes[str(p.relative_to(OUT))]=h
for p,h in protocol['checkpoint_hashes'].items():assert hashlib.sha256(Path(p.replace('/mnt/d/','D:/')).read_bytes()).hexdigest()==h
fallback_train=sum(q['fallback'] for q in trainmanifest);fallback_confirm=sum(q['fallback'] for q in manifest)
innerrows=[]
for q in frozen['candidates']:
    t=q['metrics']['mean'];innerrows.append(f"| {q['name']} | {t['regret']:.4f} | {t['PRA']*100:.2f}% | {t['T_gain']:.4f} | {'是' if q['eligible'] else '否'} |")
delta_rows='\n'.join(f'| {q["seed"]} | {b["regret"]:.4f} | {q["regret"]:.4f} | {d:.4f} |' for q,b,d in zip(summary['results']['selected']['scenes'],summary['results']['readout']['scenes'],deltas))
report=f'''# E1 混合候选、评分训练与新场景确认：整块执行报告

## 本批结论

连续完成三个大块：①固定协议与24新训练场景补采；②18+6训练场景内选择/24场景重拟合；③冻结后8新场景确认。得到真实非零残差模型 **ridge λ=1**。内部regret改善22.87%，新确认集改善仅{rel*100:.2f}%，8场景中{better}改善/{worse}退步/{ties}持平，**不足以替代readout或启动闭环**。

没有在确认结果后改模型、重选超参或换场景。按结果前固定的门槛自动跳过闭环，readout继续作为当前基线；候选模型保留用于研究。该停止由科学门槛触发，不是在等待逐步人工批准。

## 三块实际产出

| 大块 | 实际工作 | 产出 |
|---|---|---|
| A 数据与协议 | train24新场景、每场景2锚点、每状态64混合候选 | 48组/3072条真实标签；输入与监督分文件 |
| B 小模型训练 | 前18场景拟合、后6场景内部选择；零修正、两种ridge、737参数MLP | 内部拟合checkpoint、history、选择记录；最终24场景refit |
| C 冻结确认与门控 | 模型hash冻结后才生成8新确认场景 | 16组/1024条标签、确认分数与逐场景指标；闭环自动skip |

合计32新场景、64状态组、4096真实标签。每组包含逐步随机16、block随机16、readout CEM最终池16、privileged接触block16，完整池内比较；各方法选同一64候选池。CEM中间2048条预测没有真实标签，另存UNLABELED，未计入4096。

新train seed2026102900–2026102923；新confirmation seed2026103000–2026103007。与旧ranking_v2和phase21场景manifest核对seed/initial state，新train/confirmation间同样核对无重合；这不证明与未知world-model预训练数据无重合。旧test候选NPZ与test_evaluation本轮打开0。确认集本轮已经使用，后续必须视为已消耗数据，不可反复挑参数。

## 固定目标和模型

主目标仍是原physical global cost：norm([agent0xy error,Txy error])+角差权重。T-only改善只作为分项诊断及防退步约束，没有替换原任务/成功标准。

21维规划特征只来自冻结probe的预测pose/图像goal、动作均值/标准差和预测T位移/角变化。没有真实状态、真实接触、proposal标签或真实未来进入模型特征。接触候选本身来自真实几何，源状态来自CoopPolicy，因此整个实验包含privileged候选和源分布限制，不能称纯图像在线planner。

ridge拟合组内中心化的(true−readout)/100残差，修正限制±25真实代价单位。MLP固定21→32→1，零初始化输出，tanh限幅；单训练seed0、最多100epoch、patience20，pairwise softplus+0.1 MSE。normalize仅拟合集；内部选择完后用全部24训练场景重算统计/refit，确认不参与。

## 内部6场景选择

| 候选 | regret | PRA | 所选T-only gain | 达到内部资格 |
|---|---:|---:|---:|---|
{chr(10).join(innerrows)}

内部资格预先要求regret至少相对改善5%，且平均T-only gain不低于readout超过1单位；随后按global regret选择，平局优先基线/强正则/小模型。选择ridge λ=1，然后只在24训练场景refit该模型。

## 新确认8场景结果

| 指标 | readout | 固定残差模型 |
|---|---:|---:|
| 场景平均global regret ↓ | {baseline['regret']:.6f} | {selected['regret']:.6f} |
| PRA ↑ | {baseline['PRA']*100:.4f}% | {selected['PRA']*100:.4f}% |
| 所选T-only gain ↑ | {baseline['T_gain']:.6f} | {selected['T_gain']:.6f} |
| 所选global gain ↑ | {baseline['global_gain']:.6f} | {selected['global_gain']:.6f} |

| scene seed | readout regret | 残差regret | 残差−readout |
|---|---:|---:|---:|
{delta_rows}

按场景内两状态平均，再8场景等权平均；候选数不是独立样本量。只作小样本描述性确认，不给显著改善结论。T-only gain大致不变，PRA也未改善，说明新模型获益有限且未稳定覆盖多数场景。

预先闭环门槛：非零模型；确认regret改善≥10%；≥5/8场景获益；T-only不退步超过1。实际6.69%且2/8，两个条件不满足；closed_loop/summary.json明确status=skipped。闭环详细协议已在确认结果产生前冻结，未运行任何闭环episode，不报告成功率。

## 验证、算力和失败边界

全部4096候选均reset(seed)+完整prefix后分支，并逐候选第二遍native replay，state/body误差0。独立数组复核：标签重算、pose、21维特征重建、动作范围、block连续性、接触支持、分文件隔离、原权重hash全部通过。train与confirmation特征/代价重算最大误差均0。

最终ridge在train上独立重建正规方程，beta最大差3.56e-17；train-only mean/std逐元素匹配。Windows NumPy从portable checkpoint独立重算全部确认分数，最大误差{numerical_error:.2e}，逐场景regret和gate复现。没有降检查标准、跳过失败场景或改原始权重。

训练数据采集{trainmeta['wall_seconds']:.2f}秒，确认采集{confirmmeta['wall_seconds']:.2f}秒，训练选择/重拟合{frozen['training_wall_seconds']:.2f}秒；均不含脚本开发和工具启动。RTX4060用于冻结模型推理，训练小模型CPU完成；实际学习模型候选评分共{trainmeta['actual_learned_scores']+confirmmeta['actual_learned_scores']}条次。采集期PyTorch峰值allocated约{max(trainmeta['torch_peak_allocated_bytes'],confirmmeta['torch_peak_allocated_bytes'])/1024**2:.1f}MiB，不代表整机GPU峰值或最低显存。

首次ready缺失按预定t75回退：train{fallback_train}场景、确认{fallback_confirm}场景，未替换。没有新增生产依赖、没有提交或推送、没有下载新模型权重。

## 后续按大块执行的安排

1. **预测与读出诊断块**：固定训练数据，分别分析真实帧probe误差、预测latent到真实编码latent的差距、接触/非接触及T分项；据证据决定predicted-latent readout适配或阶段评分，不继续盲目扩网。
2. **单项修复与确认块**：每次只干预一个环节，训练场景内部选择，冻结后另设新确认场景；本轮确认集不再用作选择依据。只有超过同池readout且T分项不退步，才进入少量闭环。
3. **Navigation接入块**：收到组员E4仓库/模型后，一次完成环境、schema、checkpoint、readout与评估协议核查，再接现有E1组件；材料缺失会直接列给用户。本地PushT权重不代替Navigation模型。

后续默认按“诊断→实现→训练/评估→验证→归档”完整块推进；仅在缺关键材料、不可逆操作或超出已确定预算时请求输入。本批已到固定停止门槛，当前没有后台任务。

## 文件位置

protocol.json/closed_loop_protocol.json：预设协议；train_data/confirmation_data：原始数据与metadata/manifest；selected_scorer.pt和selected_scorer_portable.npz：候选模型；inner_fitted_models.pt/training_history.json/selection_frozen.json：训练审计；confirmation_scores.npz/confirmation_group_metrics.json/confirmation_summary.json：已冻结确认；各verification.json：验证。源代码归档scripts/phase22_v1。原权威Multi_Agent_LeWM.md未修改。
'''
(OUT/'MIXED_RANKER_BLOCK_REPORT.md').write_text(report,encoding='utf-8')
shutil.copy2(OUT/'MIXED_RANKER_BLOCK_REPORT.md',LOCAL/'MIXED_RANKER_BLOCK_REPORT.md')
card='''# Mixed ranker v1 数据与加载边界

train24场景，内部fit18/selection6；confirmation8场景在模型冻结后才采集。本轮confirmation已消耗，不重新调参。

每组64候选固定顺序：0–15 random_step、16–31 random_block、32–47 readout_CEM、48–63 privileged_contact_block。所有方法使用同一池。按scene分区，禁止按候选拆train/test。

inputs.npz中features[64,21]是唯一小模型特征，predicted_cost[64]为基线；其余规划数组为预测latent、goal latent、probe pose、动作和像素。proposal_id只供分组审计，不是模型输入。

labels.npz保存真实未来state/contact、true_cost、true_T_cost和real_encoded_*等监督，只能用于训练目标或评估；不得进入在线特征。真实初态、goal、direction也是监督/privileged诊断键。

cem_intermediate_UNLABELED.npz为前两轮CEM预测，共2048条，没有真实标签，不计入4096标签。source.npz仅用于仿真复现。

可移植ridge评分：readout_cost/100 + clip(((features-mean)/std) @ beta, -0.25, 0.25)。features必须遵循phase22_common.low_features及feature_names，不能用Navigation输入替代。正式checkpoint须配套protocol/hash/冻结probe；候选模型未通过闭环启用门槛，不能宣称优于readout的已验证闭环模型。
'''
(OUT/'DATASET_CARD.md').write_text(card,encoding='utf-8')
writeobj=dict(status='passed',independent_confirmation_score_max_error=numerical_error,scene_regret_recomputed=True,
    closed_loop_gate_recomputed=True,closed_loop_skipped=True,train_confirmation_seeds_disjoint=True,
    hash_checked_input_label_files=len(hashes),original_weights_unchanged=True,old_test_candidate_npz_opened=0)
(OUT/'final_verification.json').write_text(json.dumps(writeobj,indent=2),encoding='utf-8')
(OUT/'dataset_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
archive=ROOT/'scripts/phase22_v1';shutil.copy2(Path(__file__),archive/Path(__file__).name)
sourcehash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in archive.iterdir() if p.is_file()}
for p in archive.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
(OUT/'source_sha256.json').write_text(json.dumps(sourcehash,indent=2),encoding='utf-8')
entry=f'''

## E1 混合候选大块执行完成（2026-10-24）

用户要求后续按大块自主推进，已连续完成冻结协议→24新训练场景补采→18/6内部选型→24重拟合→冻结后8新确认场景，共4096标签；输入/监督分文件，全部二次full-prefix native重放0，独立特征/标签/动作/hash检查通过，旧test候选打开0。

21维小模型候选为zero、ridge.01/1、737参数MLP；内部选中ridge1，regret8.5919→6.6266，T-only9.2379→12.2803。新确认8场景regret{baseline['regret']:.4f}→{selected['regret']:.4f}(改善{rel*100:.2f}%)，2优/1差/5平，PRA基本不变，T-only18.9059→18.8990。未达到预设闭环门槛(>=10%、>=5/8获益)，自动跳过闭环，继续readout基线，候选checkpoint保留。不是等待人工批准，未在确认后调整模型。

最终ridge独立正规方程beta误差3.56e-17，Windows portable checkpoint重算确认得分误差{numerical_error:.2e}，gate复算通过。无权重修改/新依赖/提交/推送，无后台任务。报告results/e1/mixed_ranker_block_v1/MIXED_RANKER_BLOCK_REPORT.md，源scripts/phase22_v1。

后续默认按完整诊断/修复/确认块执行，不频繁索要下一步指令；缺材料、不可逆操作或超预算才请求输入。优先分离接触预测与probe误差，再定单项适配；本确认集已消耗，后续需新确认集。Navigation仍待组员E4材料，不用PushT替代。
'''
state=ROOT/'PROJECT_STATE.md';assert 'E1 混合候选大块执行完成' not in state.read_text(encoding='utf-8')
with state.open('a',encoding='utf-8') as f:f.write(entry)
shutil.copy2(state,LOCAL/'PROJECT_STATE.md')
print(json.dumps(writeobj,indent=2))
