"""Export ONLY our small trained direct scorer for teammate inference/reproduction."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import state_action_ranker as sr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repair',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise RuntimeError('Preserve previous exported model')
    frozen=json.loads((a.repair/'selection_frozen.json').read_text());checkpoint=a.repair/'invariant.pt';assert sr.sha(checkpoint)==frozen['models']['invariant']['checkpoint_sha256']
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    with np.load(a.repair/'invariant_portable.npz') as z:
        arrays={k:z[k].astype(np.float32).tolist() for k in z.files if k not in ['cost_mean','cost_std']}
    obj=dict(schema='direct_safety_mlp_v1',kind='direct current-state/action scorer; NOT Navigation LeWM',architecture=[44,64,32,3],
        parameter_count=5059,epoch=c['epoch'],policy=c['policy'],cost_mean=c['cost_mean'],cost_std=c['cost_std'],
        source_checkpoint_sha256=sr.sha(checkpoint),source_portable_npz_sha256=sr.sha(a.repair/'invariant_portable.npz'),
        ownership='Locally trained scorer from safety_repair_v1; this exports no teammate LeWM/PushT checkpoint.',
        limitation='Finite two-agent H8 direct scorer. Risk estimates can be wrong; no safety certification. Inference-only frozen artifact.',arrays=arrays)
    a.output.parent.mkdir(parents=True,exist_ok=True);sr.write(a.output,obj);print(sr.sha(a.output))
if __name__=='__main__':main()
