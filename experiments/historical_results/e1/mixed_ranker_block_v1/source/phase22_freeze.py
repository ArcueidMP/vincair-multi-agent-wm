"""Write all block decisions before collecting training data."""
from pathlib import Path
import datetime,shutil
from phase22_common import ROOT,OUT,ENV,ARMS,FEATURE_NAMES,sha,write
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
weights=ROOT/'data/checkpoints/lewm_coop2_joint'
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    train_seeds=list(range(2026102900,2026102924)),fit_scene_indices=list(range(18)),inner_selection_scene_indices=list(range(18,24)),
    confirmation_seeds=list(range(2026103000,2026103008)),environment=ENV,
    candidates=dict(arms=ARMS,each=16,mixed_pool=64,horizon=10,anchors='t0 and first ready in1..140; fallback75; source160; no replacements',
        cem='16x3 elite4 stdfloor.05, original global objective; final pool only labeled',contact='True geometry privileged, speedU[0,.3], two5step chunks, angleU[-atan.25,atan.25]'),
    feature_names=FEATURE_NAMES,primary='Scene-average top1 regret under original physical global cost',
    secondary=['PRA global','selected T-only gain','T-only oracle and opportunity gap','per-proposal metrics','real-encoded probe diagnostic'],
    models=['zero_correction','ridge_lambda_0.01','ridge_lambda_1','residual_mlp_21_32_1'],
    ridge='Group-centered residual ridge, normalized MSE+lambda*||beta||^2; clip correction ±0.25 scaled units',
    mlp=dict(seed=0,epochs_max=100,patience=20,learning_rate=.001,weight_decay=.01,batch_states=8,
        loss='Within-state pairwise softplus abs true difference>1 +0.1 scaled cost MSE',correction='.25*tanh(network)',
        selection='Minimum inner scene-average global regret, earlier epoch breaks ties; refit24 using fixed selected epoch count.'),
    normalization='Fit18 train only for selection; refit24 for final; std floor.01',
    selection='Minimum inner global regret. Ties within1e-8: zero, ridge1, ridge.01, MLP. Learned model eligible only if inner relative improvement>=5% and mean selected T-only gain no worse than baseline by>1.',
    confirmation='Generate after selection_frozen/checkpoint hash. Evaluate fixed selected scorer and same-pool readout once; no refit after confirmation.',
    closed_loop_gate='Selected nonzero AND confirmation regret improves>=10% AND better in>=5/8 scenes AND mean selected T-only gain not worse by>1. Otherwise skip closed-loop.',
    closed_loop_if_gate='At most4 separate new seeds2026103100..03 x2 methods,150steps; same privileged coordinator/proposals; offline improvement not success claim.',
    unknown_wm_pretrain_overlap=True,old_test_candidate_access=False,
    checkpoint_hashes={str(p):sha(p) for p in [weights/'weights.pt',weights/'config.json',weights/'readout/cost_probe.pt']},
    limitations=['All mixed pools include privileged geometry candidates; no pure image-only planning claim.',
        'New confirmation becomes consumed after evaluation, never silently reused for tuning.',
        'Fixed generated sources from privileged CoopPolicy differ from learned closed-loop.'])
write(OUT/'protocol.json',protocol)
archive=ROOT/'scripts/phase22_v1';archive.mkdir(parents=True,exist_ok=True)
for name in ['phase22_common.py','phase22_freeze.py','e1_ranker_common.py']:
    p=Path(__file__).with_name(name);shutil.copy2(p,archive/name)
print('PROTOCOL FROZEN',OUT)
