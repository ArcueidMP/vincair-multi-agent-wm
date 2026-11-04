"""The pre-registered analysis of ANALYSIS.md, Sections 4 and 5, on the `test` results.

    python -m uav_arena.analysis --results data/uav_arena/results

J comes from each model's `test` records (`run.py`); L1-L3 come from the harness's per-question arrays
(`harness/<model>_test.npz`), computed as the harness's summary computes them. Before any statistic, every model
must have one episode per layout and condition and the same questions in the same order, its success counts must
equal its run's per-condition summary, and its recomputed layers must equal the harness summary, overall and per
condition.

Intervals are 95% percentile intervals from 10,000 bootstrap resamples of the 50 `test` layouts: the multinomial
layout counts from NumPy `default_rng(0)` that README's results use, so every interval comes from the same
resamples. A resampled layout brings its 8 episodes and its questions, and every model is recomputed on the same
resamples, so models and layers stay paired; the set of models is fixed. Spearman correlations rank the models
with average ranks for ties. The numbers go to `<results>/primary_analysis.json`; the report is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from .harness import CHANGE_KIND
from .scenarios import CONDITIONS as _CONDITIONS

RESAMPLES = 10_000
CONDITIONS = [c.name for c in _CONDITIONS]
ALL = tuple(range(len(CONDITIONS)))
A_CHANGES = CHANGE_KIND == 0
ROWS = dict(full="full", no_cross="no_cross", flat="flat", war="W-ar", pets="PETS")
PRIMARY = [(row, n, s) for row in ROWS for n in (2000, 500) for s in (1, 2, 3)]
REFERENCES = {"physics": ("physics_mppi_test", "physics_test"),
              "physics + mass estimation": ("adaptive_mppi_test", "adaptive_test"),
              "example MLP": ("example_mlp_test", "example_mlp_test")}
MODELS = PRIMARY+list(REFERENCES)
PRIMARY_SET, SENSITIVITY_SET = list(range(len(PRIMARY))), list(range(len(MODELS)))
RUN_PROVENANCE = dict(full="jepa", no_cross="jepa", flat="jepa", war="war", pets="pets")   # <name>_test_provenance.json
LAYERS = dict(L1="L1", L2="L2", L3="L3", L2_payload="L2_A_payload", L2_others="L2_A_others", L2_own="L2_A_own")
METRICS = ("J", *LAYERS)


def label(model):
    return model if isinstance(model, str) else f"{ROWS[model[0]]} {model[1]} s{model[2]}"


def stems(model):
    """The closed-loop result's and the harness result's file stems."""
    if isinstance(model, str):
        return REFERENCES[model]
    row, n, s = model
    if row == "pets":
        return f"pets_e{n}_s{s}_mppi_test", f"pets_e{n}_s{s}_test"
    return (f"formal_{row}_h4_60k_ep{n}_s{s}_test",)*2


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- statistics ----------
def spearman(*columns):
    """Spearman correlation matrices of columns of (resamples, models) values, per resample: (resamples, k, k)."""
    ranks = np.stack([rankdata(c, axis=1) for c in columns], 1)
    ranks -= ranks.mean(2, keepdims=True)
    ranks /= np.sqrt((ranks**2).sum(2, keepdims=True))
    return np.einsum("bkm,blm->bkl", ranks, ranks)


def partial(C):
    """The correlation of variables 0 and 1 given the others, from correlation matrices."""
    P = np.linalg.inv(C)
    return -P[:, 0, 1]/np.sqrt(P[:, 0, 0]*P[:, 1, 1])


def partial_formula(C):
    """ANALYSIS.md, Section 4: rho(x, y | z) from the three correlations among x, y and z."""
    rxy, rxz, rzy = C[:, 0, 1], C[:, 0, 2], C[:, 2, 1]
    return (rxy-rxz*rzy)/np.sqrt((1-rxz**2)*(1-rzy**2))


def interval(samples):
    """The sample's value (row 0) and the 95% percentile interval over the resamples (the other rows)."""
    boot = samples[1:]
    out = dict(estimate=float(samples[0]), low=float(np.nanpercentile(boot, 2.5)),
               high=float(np.nanpercentile(boot, 97.5)))
    if np.isnan(boot).any():
        out["undefined_resamples"] = int(np.isnan(boot).sum())
    return out


def position(iv):
    return "below 0" if iv["high"] < 0 else "above 0" if iv["low"] > 0 else "contains 0"


def verdict(iv, distinct_j):
    """ANALYSIS.md, Section 4's decision rule."""
    if distinct_j < 5:
        return "uninformative"
    return {"below 0": "supported", "contains 0": "not established", "above 0": "unexpected direction"}[position(iv)]


# ---------- data ----------
class Analysis:
    """Every model's `test` results, checked against their own summaries, with each metric on the sample and on
    every resample."""

    def __init__(self, results):
        self.results = Path(results).absolute()
        self.runs, self.scores, questions = [], [], []
        for model in MODELS:
            run, harness = stems(model)
            self.runs.append(json.loads((self.results/f"{run}.json").read_text()))
            self.scores.append(json.loads((self.results/"harness"/f"{harness}.json").read_text()))
            with np.load(self.results/"harness"/f"{harness}.npz") as f:
                questions.append({key: f[key] for key in ("seed", "condition", "decision", "l1_payload", "l2_error",
                                                            "l3_costly", "l3_reliable")})
        self.layouts = sorted({r["seed"] for r in self.runs[0]["records"]})
        first = questions[0]
        layout = np.searchsorted(self.layouts, first["seed"])
        condition = np.array([CONDITIONS.index(c) for c in first["condition"]])
        reliable = first["l3_reliable"].astype(bool)
        if len(self.layouts) != 50 or not np.array_equal(np.asarray(self.layouts)[layout], first["seed"]):
            raise ValueError("the questions' layouts are not the 50 layouts of the runs")
        for model, q in zip(MODELS, questions):
            if not all(np.array_equal(q[key], first[key]) for key in ("seed", "condition", "decision", "l3_reliable")):
                raise ValueError(f"{label(model)} was scored on other questions")
        manifests = {s["provenance"]["states_manifest_sha256"] for s in self.scores}
        if len(manifests) != 1:
            raise ValueError(f"the harness results come from {len(manifests)} question sets")
        self.manifest, self.questions, self.reliable = manifests.pop(), len(layout), int(reliable.sum())

        def cells(values):
            """Per-question values -> their sums per layout and condition (50, 8)."""
            out = np.zeros((len(self.layouts), len(CONDITIONS)))
            np.add.at(out, (layout, condition), values)
            return out

        shape = (len(MODELS), len(self.layouts), len(CONDITIONS))
        self.num, episodes = {metric: np.zeros(shape) for metric in METRICS}, np.zeros(shape)
        for i, (run, q) in enumerate(zip(self.runs, questions)):
            for r in run["records"]:
                cell = (i, self.layouts.index(r["seed"]), CONDITIONS.index(r["condition"]))
                episodes[cell] += 1
                self.num["J"][cell] += r["outcome"] == "success"
            error = q["l2_error"][:, A_CHANGES]
            self.num["L1"][i] = cells(q["l1_payload"].mean(1))
            self.num["L2"][i] = cells((0.5*error[..., 0]+0.5*error[..., 1]).mean(1))
            for p, part in enumerate(("payload", "others", "own")):
                self.num[f"L2_{part}"][i] = cells(error[..., p].mean(1))
            self.num["L3"][i] = cells(q["l3_costly"]*reliable)
        if not (episodes == 1).all():
            raise ValueError("a model lacks or repeats an episode")
        self.den = dict(J=np.ones(shape[1:]), L3=cells(reliable.astype(float)))
        self.per_question = cells(np.ones(len(layout)))
        counts = np.random.default_rng(0).multinomial(len(self.layouts), np.full(len(self.layouts), 1/len(self.layouts)),
                                                      size=RESAMPLES)
        self.weights = np.vstack([np.ones(len(self.layouts)), counts])     # layout counts; row 0 is the sample
        self._values = {}
        self.check()

    def value(self, metric, conds=ALL):
        """A metric of every model on the sample (row 0), then on each resample: (1 + RESAMPLES, models)."""
        key = (metric, tuple(conds))
        if key not in self._values:
            num = self.num[metric][:, :, list(conds)].sum(2)
            den = self.den.get(metric, self.per_question)[:, list(conds)].sum(1)
            self._values[key] = (self.weights @ num.T)/(self.weights @ den)[:, None]
        return self._values[key]

    def check(self):
        """Success counts against the runs' summaries, layers against the harness's, overall and per condition."""
        for i, (model, run, score) in enumerate(zip(MODELS, self.runs, self.scores)):
            for entry in run["summary"]:
                if self.num["J"][i, :, CONDITIONS.index(entry["condition"])].sum() != entry["success"]:
                    raise ValueError(f"{label(model)}: successes differ from the run's summary in {entry['condition']}")
            for key, conds in [("all", ALL)]+[(c, (j,)) for j, c in enumerate(CONDITIONS)]:
                for metric, field in LAYERS.items():
                    ours, theirs = self.value(metric, conds)[0, i], score["summary"][key][field]
                    if not np.isclose(ours, theirs, rtol=1e-9, atol=0):
                        raise ValueError(f"{label(model)}: {metric} in {key} is {ours}, the harness says {theirs}")

    def provenance(self, i):
        """ANALYSIS.md, Section 6: the harness commit, the question set and the checkpoint behind a model's result."""
        model, run, score = MODELS[i], self.runs[i], self.scores[i]
        out = dict(score["provenance"])
        checkpoint = json.loads(run["config"]["predictor_options"] or "{}").get("checkpoint")
        scored = json.loads(score["config"]["predictor_options"] or "{}").get("checkpoint")
        if checkpoint != scored:
            raise ValueError(f"{label(model)}: the closed loop flew {checkpoint}, the harness scored {scored}")
        if checkpoint:
            path = self.results.parents[2]/checkpoint     # paths are relative to the checkout above data/
            out.update(checkpoint=checkpoint,
                       checkpoint_here_matches=sha256(path) == out["checkpoint_sha256"] if path.exists() else None)
        name = "test_run" if isinstance(model, str) else f"{RUN_PROVENANCE[model[0]]}_test"
        source = self.results/f"{name}_provenance.json"
        out["closed_loop_source"] = json.loads(source.read_text()).get("source") if source.exists() else None
        return out

    def correlations(self, select, conds=ALL):
        """The primary statistic and the correlations of Section 5 on the models `select` and the conditions `conds`."""
        v = {metric: self.value(metric, conds)[:, select] for metric in METRICS}
        C = spearman(v["L2"], v["J"], v["L1"], v["L3"], v["L2_own"], v["L2_payload"], v["L2_others"])
        sub = lambda *k: C[:, list(k)][:, :, list(k)]
        primary = partial_formula(sub(0, 1, 2))
        if not np.allclose(primary, partial(sub(0, 1, 2)), equal_nan=True):
            raise ValueError("the two forms of the partial correlation disagree")
        return {"distinct J": len(np.unique(self.num["J"][select][:, :, list(conds)].sum((1, 2)))),
                "rho(L2, J | L1)": interval(primary),
                "rho(L3, J | L1)": interval(partial_formula(sub(3, 1, 2))),
                "rho(L1, J)": interval(C[:, 2, 1]), "rho(L2, J)": interval(C[:, 0, 1]),
                "rho(L3, J)": interval(C[:, 3, 1]), "rho(L2, L1)": interval(C[:, 0, 2]),
                "rho(L2, J) - rho(L1, J)": interval(C[:, 0, 1]-C[:, 2, 1]),
                "rho(L2, J | L1, own)": interval(partial(sub(0, 1, 2, 4))),
                "rho(L2 payload, J | L1)": interval(partial_formula(sub(5, 1, 2))),
                "rho(L2 others, J | L1)": interval(partial_formula(sub(6, 1, 2))),
                "rho(L2 payload, J)": interval(C[:, 5, 1]), "rho(L2 others, J)": interval(C[:, 6, 1])}

    def difference(self, metric, pairs, conds=ALL):
        """The mean over pairs of models (a, b) of metric(a) - metric(b)."""
        v = self.value(metric, conds)
        return interval(np.mean([v[:, MODELS.index(a)]-v[:, MODELS.index(b)] for a, b in pairs], 0))

    def summary(self):
        primary = self.correlations(PRIMARY_SET)
        primary["verdict"] = verdict(primary["rho(L2, J | L1)"], primary["distinct J"])
        return dict(
            analysis="ANALYSIS.md, Sections 4 and 5 (arena-v0.3)", resamples=RESAMPLES,
            bootstrap="numpy default_rng(0).multinomial(50, [1/50]*50, size=10000): layout counts per resample",
            layouts=self.layouts, conditions=CONDITIONS, questions=self.questions, reliable_questions=self.reliable,
            states_manifest_sha256=self.manifest,
            models={label(m): dict(closed_loop=f"{stems(m)[0]}.json", harness=f"harness/{stems(m)[1]}.json",
                                   provenance=self.provenance(i),
                                   all={metric: interval(self.value(metric)[:, i]) for metric in METRICS},
                                   conditions={c: {metric: interval(self.value(metric, (j,))[:, i]) for metric in METRICS}
                                               for j, c in enumerate(CONDITIONS)})
                    for i, m in enumerate(MODELS)},
            primary_set=dict(models=[label(MODELS[i]) for i in PRIMARY_SET], **primary),
            per_condition={c: self.correlations(PRIMARY_SET, (j,)) for j, c in enumerate(CONDITIONS)},
            per_condition_means={c: {metric: float(self.value(metric, (j,))[0, PRIMARY_SET].mean())
                                     for metric in ("J", "L1", "L2", "L3")} for j, c in enumerate(CONDITIONS)},
            data_size={row: {metric: self.difference(metric, [((row, 2000, s), (row, 500, s)) for s in (1, 2, 3)])
                             for metric in ("J", "L1", "L2", "L3")} for row in ROWS},
            structure={f"{a} - {b}": {"+".join(map(str, ns)): {metric: self.difference(
                metric, [((a, n, s), (b, n, s)) for n in ns for s in (1, 2, 3)]) for metric in ("J", "L2")}
                for ns in ((2000,), (500,), (2000, 500))}
                for a, b in (("full", "no_cross"), ("full", "flat"), ("no_cross", "flat"))},
            sensitivity_set=dict(models=[label(MODELS[i]) for i in SENSITIVITY_SET],
                                 **self.correlations(SENSITIVITY_SET)))


# ---------- report ----------
def show(iv, scale=1.0, digits=3, unit="", sign=""):
    f = lambda x: f"{scale*x:{sign}.{digits}f}"
    return f"{f(iv['estimate'])}{unit} ({f(iv['low'])} to {f(iv['high'])})"


def report(out):
    def correlations(c, keys):
        for key in keys:
            print(f"  {key:26s} {show(c[key], sign='+')}  [{position(c[key])}]")

    primary = out["primary_set"]
    print(f"arena-v0.3 `test`: ANALYSIS.md, Sections 4-5; {out['resamples']} resamples of the "
          f"{len(out['layouts'])} layouts")
    print(f"{out['questions']} questions ({out['reliable_questions']} with a reliable reference), states manifest "
          f"{out['states_manifest_sha256'][:12]}")
    print("checks passed: one episode per layout and condition, identical questions, success counts = run summaries, "
          "L1-L3 and L2's parts = harness summaries (overall and per condition)")
    print("\n== Models: J, L1 and L2 (cm), L3, with 95% intervals over layouts")
    for name, model in out["models"].items():
        t = model["all"]
        print(f"{name:26s} J {show(t['J'], 100, 2, '%')}  L1 {show(t['L1'], 100, 2)}  "
              f"L2 {show(t['L2'], 100, 2)}  L3 {show(t['L3'], 100, 1, '%')}")
    print(f"\n== Primary test, {len(primary['models'])} models: rho(L2, J | L1) = "
          f"{show(primary['rho(L2, J | L1)'], sign='+')}")
    print(f"  verdict: {primary['verdict']} (J takes {primary['distinct J']} distinct values; fewer than 5 is "
          f"uninformative)")
    correlations(primary, ("rho(L2, J)", "rho(L2, L1)", "rho(L1, J)"))
    print("\n== Secondary analyses (no decision rule)")
    correlations(primary, ("rho(L3, J | L1)", "rho(L3, J)", "rho(L2, J) - rho(L1, J)", "rho(L2, J | L1, own)",
                           "rho(L2 payload, J | L1)", "rho(L2 others, J | L1)", "rho(L2 payload, J)",
                           "rho(L2 others, J)"))
    print("\n== Per condition, 30 models: mean J, L1, L2 (cm), L3 over the models; distinct J; correlations")
    for c, pc in out["per_condition"].items():
        mean = out["per_condition_means"][c]
        print(f"{c:20s} J {100*mean['J']:4.1f}%  L1 {100*mean['L1']:5.2f}  L2 {100*mean['L2']:4.2f}  "
              f"L3 {100*mean['L3']:4.1f}%  distinct J {pc['distinct J']:2d}")
        correlations(pc, ("rho(L2, J | L1)", "rho(L3, J | L1)", "rho(L1, J)", "rho(L2, J)"))
    print("\n== Data size: 2,000 minus 500 episodes, mean over the 3 seeds (J and L3 in points, L1 and L2 in cm)")
    for row, d in out["data_size"].items():
        print(f"{ROWS[row]:9s} J {show(d['J'], 100, 2, sign='+')}  L1 {show(d['L1'], 100, 2, sign='+')}  "
              f"L2 {show(d['L2'], 100, 2, sign='+')}  L3 {show(d['L3'], 100, 1, sign='+')}")
    print("\n== Structure: differences paired by seed and data size (J in points, L2 in cm)")
    for pair, by_size in out["structure"].items():
        for sizes, d in by_size.items():
            print(f"{pair:18s} {sizes:9s} J {show(d['J'], 100, 2, sign='+')}  L2 {show(d['L2'], 100, 2, sign='+')}")
    sensitivity = out["sensitivity_set"]
    print(f"\n== Sensitivity set, {len(sensitivity['models'])} models (+ physics, physics + mass estimation, "
          f"example MLP)")
    print(f"  distinct J {sensitivity['distinct J']}")
    correlations(sensitivity, ("rho(L2, J | L1)", "rho(L3, J | L1)", "rho(L1, J)", "rho(L2, J)", "rho(L3, J)"))
    mismatch = [name for name, model in out["models"].items()
                if model["provenance"].get("checkpoint_here_matches") is False]
    print(f"\nharness commits: {sorted({m['provenance']['source']['describe'] for m in out['models'].values()})}; "
          f"checkpoints whose file here differs from the harness's hash: {mismatch or 'none'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="the directory of run.py and harness results")
    parser.add_argument("--out", type=Path, default=None, help="default: <results>/primary_analysis.json")
    args = parser.parse_args()
    out = Analysis(args.results).summary()
    (args.out or args.results/"primary_analysis.json").write_text(json.dumps(out, indent=1)+"\n")
    report(out)


if __name__ == "__main__":
    main()
