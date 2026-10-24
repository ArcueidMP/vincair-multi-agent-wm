"""Teammate-facing comparison against published episode outcomes, not latency."""
import argparse,json
from pathlib import Path
import numpy as np
import state_action_ranker as sr

METRICS=['joint_success','safe_joint_success','substep_loss_of_separation','substep_collision_proxy','steps','substep_minimum_separation']
def compare(expected,actual,tolerance=1e-6):
    def key(r):return (r['geometry'],r['arm'],r['seed'])
    reference={key(r):r for r in expected};observed={key(r):r for r in actual}
    if len(reference)!=len(expected) or len(observed)!=len(actual):raise ValueError('Duplicate geometry/arm/seed rows')
    if set(reference)!=set(observed):raise ValueError('Missing or extra episode keys')
    errors=[]
    for k,r in reference.items():
        q=observed[k]
        for metric in METRICS:
            ok=abs(q[metric]-r[metric])<=tolerance if metric=='substep_minimum_separation' else q[metric]==r[metric]
            if not ok:errors.append(dict(key=list(k),metric=metric,reference=r[metric],actual=q[metric]))
    return errors

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--original-raw',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise RuntimeError('Preserve previous comparison')
    expected=json.loads((a.reference/'episode_results.json').read_text());actual=json.loads((a.run/'results.json').read_text());errors=compare(expected,actual)
    rp=json.loads((a.run/'reproduction_protocol.json').read_text());verify=json.loads((a.reference/'verification.json').read_text())
    assert rp['model_json_sha256']==verify['portable_json_sha256'] and rp['original_protocol_sha256']==sr.sha(a.reference/'protocol.json')
    assert rp['full'] and rp['tasks']==len(expected)==len(actual)==72
    arraychecks=0;rawhashes={}
    for row in actual:
        name=f"{row['geometry']}_{row['arm']}_{row['seed']}";p=a.run/name/'trajectory.npz';rawhashes[str(p.relative_to(a.run))]=sr.sha(p)
        assert row['state_replay_error']==row['substep_replay_error']==0
        if a.original_raw:
            with np.load(p) as x,np.load(a.original_raw/'episodes'/name/'trajectory.npz') as y:
                for field in ['state','substep_state','action','goals','selected','fallback','candidates','prediction','oracle_cost','oracle_minimum']:
                    np.testing.assert_array_equal(x[field],y[field]);arraychecks+=1
    result=dict(status='passed' if not errors else 'mismatch',episodes=len(actual),metric_differences=errors,latency_compared=False,
        exact_array_fields_compared=arraychecks,model_and_protocol_hash_match=True,all_native_replay_errors_zero=True,
        scope='Compares the supplied run against published artifacts; machine/environment independence is not inferred by this tool')
    a.output.mkdir(parents=True);sr.write(a.output/'verification.json',result);sr.write(a.output/'trajectory_sha256.json',rawhashes);print(json.dumps(result))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
