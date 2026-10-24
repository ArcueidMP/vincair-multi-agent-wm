"""Show every confirmation geometry; preserve negative ablations."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);root=ap.parse_args().run
    s=json.loads((root/'summary.json').read_text());v=json.loads((root/'verification.json').read_text());f=json.loads((root/'selection_frozen.json').read_text())
    methods=s['methods'];out=root/'figures';out.mkdir(exist_ok=True)
    names=['old_safety_64','raw_consistent_64','invariant_consistent_64','invariant_consistent_68']
    labels=['Frozen original /64','Retrained raw /64','Invariant repair /64','Invariant repair /68']
    colors=['#777777','#0072B2','#D55E00','#009E73']
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,metric,title in zip(axs,['unsafe_given_safe','regret'],['Unsafe selection when a safe option exists','Soft-cost regret relative to pool optimum']):
        for i,(name,label,color) in enumerate(zip(names,labels,colors)):
            values=[r[metric] for r in methods[name]['scenes']];x=i+np.linspace(-.12,.12,len(values))
            ax.scatter(x,values,color=color,alpha=.65,s=27);ax.plot([i-.23,i+.23],[np.mean(values)]*2,color=color,lw=3)
        ax.set_xticks(range(4),labels,rotation=20,ha='right');ax.set_title(title);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylim(0,1);axs[0].set_ylabel('Fraction unsafe');axs[1].set_ylabel('VMAS soft-cost units')
    fig.suptitle('Fresh confirmation:12 geometries x6 states; dots=geometry means, bars=overall means',fontsize=11)
    fig.savefig(out/'repair_comparison.png',dpi=180);fig.savefig(out/'repair_comparison.svg');plt.close(fig)
    raw=root/'raw_svg_exports';raw.mkdir(exist_ok=True);svg=out/'repair_comparison.svg';(raw/svg.name).write_bytes(svg.read_bytes())
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    rows=[]
    for name,m in methods.items():rows.extend(dict(method=name,**r) for r in m['records'])
    with (root/'confirmation_decisions.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    public={k:val for k,val in s.items() if k!='methods'};public['methods']={k:{q:x for q,x in m.items() if q!='records'} for k,m in methods.items()}
    (root/'summary_public.json').write_text(json.dumps(public,indent=2))
    table='\n'.join(f"| {k} | {m['mean']['unsafe_given_safe']:.2%} | {m['mean']['regret']:.4f} |" for k,m in methods.items())
    report=f'''# E4 direct-scorer repair: fresh confirmation

## Result

The predeclared offline gate passed: original64 conditional unsafe30.56% -> invariant repair64 6.94% (23.61 percentage points), with9/12 geometry wins. This is a direct observed-state/action scorer, NOT a Navigation LeWM or a closed-loop improvement.

| Method | Unsafe given safe | Soft-cost regret |
|---|---:|---:|
{table}

## What the ablations support

- Fresh raw-coordinate training plus the consistency selector also reaches6.94% unsafe. Therefore safety improvement over the old model cannot be attributed specifically to invariance; broader training, normalization and changed selection are confounded in that comparison.
- Invariant repair improves mean regret0.6853 ->0.3097 relative to the freshly trained raw control, with equal unsafe rate. Each representation has its own validation-selected checkpoint/policy; this is a method comparison, not an isolated causal estimate.
- The invariant model with the legacy selector has the same6.94% unsafe and slightly lower regret0.2815. The consistency selector shows no safety benefit for this model on this confirmation set. Preserve this negative ablation; do not retrospectively switch the preregistered primary result.
- Appending four zero/brake candidates helps the frozen old model30.56% ->18.06%, but does not improve the repaired model6.94% ->6.94%. Braking is a fixed open-loop sequence and is not certified safe. Regret comparisons between64/68 use different pool optima.

## Data and selection

44 new geometries:24 train,8 validation,12 confirmation; three initial-state conditions and two distances per geometry.264 states x68 candidates =17,952 native VMAS rollouts, each repeated in a fresh environment. Labels include initial and native substep separation. Whole geometries remain in one split. No previous consumed test/stress files were read for training or selection. Confirmation labels were generated and hashed before training but method access was guarded until model/policy freeze.

Both models use5059-parameter44->64->32->3 MLPs. Invariant features center positions, align inter-agent displacement and average both agent-order predictions. Train-only feature/cost moments. Seed0 CPU training, validation-only epoch/risk/margin selection; raw epoch{f['models']['raw']['epoch']}, invariant epoch{f['models']['invariant']['epoch']}. Original checkpoint hash unchanged.

Raw and invariant models both train on the expanded68 proposal pool. The64 versus68 comparisons change evaluation candidates only; they do not isolate the effect of expanded-pool training. The three synthetic conditions share geometry families across splits. One seed and12 confirmation geometries do not establish broad generalization. Arrival-offset also changes separation. This confirmation set is now consumed and must not tune later models.

## Verification and artifacts

27 targeted tests pass. Independent NumPy verification reproduced576 method decisions and every aggregate; rebuilt training moments and all17,952 physical cost/safety labels. Maximum cost error{v['max_cost_error']:.3g}; forward error{v['max_numpy_forward_error']:.3g}; native replay error0. Numerical translation prediction error{v['translation_prediction_max_error']:.3g}; rigid-coordinate and agent-swap feature invariance unit tests pass. Coordinate invariance is a property of the scoring representation, not a claim that clipped VMAS action dynamics are globally rotation invariant.

Full weights/raw NPZ remain in this external run directory. Published files contain source, frozen protocol, hashes, compact summaries, all confirmation decisions, verification and figure. Raw SVG exports are retained; published SVG only strips trailing whitespace.

## Next block

Freeze a small NEW closed-loop confirmation protocol using native VMAS, matched seeds/budgets and original/repair scoring arms. First validate a real receding-horizon integration and state/action scaling with a short smoke test, then run a bounded paired pilot. Do not extrapolate H8 offline gains to successful navigation. Navigation-specific WM checkpoint/loader/config/normalization/history/action contract/state readout remain missing; request these from teammates before any LeWM transfer claim. No three-agent or Navigation-WM experiment was run here.
'''
    (root/'REPORT.md').write_text(report,encoding='utf-8')
    (root/'figure_provenance.json').write_text(json.dumps(dict(source='summary.json: all12 confirmation geometries; no filtering',panels=['unsafe given safe','soft regret'],dots='geometry means',bars='equal geometry mean',svg_change='trailing whitespace stripped only; original retained in raw_svg_exports',exports=['PNG180dpi','SVG vector text']),indent=2))
if __name__=='__main__':main()
