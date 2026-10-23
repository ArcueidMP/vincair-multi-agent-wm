"""Download checkpoints into $STABLEWM_HOME/checkpoints (default ./data/checkpoints).

    python scripts/download.py                 # official LeWM Push-T + the two coop2 models + the two VMAS models (~220 MB)
    python scripts/download.py --vmas-data     # also the VMAS data: train / heldout / aggregated train (~300 MB)
    python scripts/download.py --coop-heldout  # also the held-out coop scenes the readouts were fit on (700 MB)
    python scripts/download.py --pusht-data    # also the official Push-T dataset (44 GB, only to re-export assets/)

Layout:
    checkpoints/lewm/pusht/{weights.pt,config.json}          official single-agent LeWM (quentinll/lewm-pusht)
    checkpoints/lewm_coop2_ftfull/{weights.pt,config.json}   fine-tuned on coop data, own action
    checkpoints/lewm_coop2_joint/{weights.pt,config.json}    fine-tuned on coop data, action = [self, partner]
    checkpoints/lewm_coop2_*/readout/cost_probe.pt           state readout of each coop2 model (readout.py)
    datasets/multipusht_2a_coop_heldout.h5                   held-out coop scenes (--coop-heldout)
    vmas_pusht/wm_full_v2.pt                                 VMAS entity-token LeWM, full attention
    vmas_pusht/wm_full_h_agg2.pt                             ... with a jointly trained state readout head
    vmas_pusht/{train,heldout,train_agg2}.pt                 VMAS data (--vmas-data)
"""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parents[1] / "data"))

from huggingface_hub import hf_hub_download, list_repo_files

LEWM_REPO = "quentinll/lewm-pusht"
COOP_REPO = os.environ.get("COOP_REPO", "fuisloy/vincair-multi-agent-wm")  # Hugging Face model repo with the released checkpoints
COOP_MODELS = ("lewm_coop2_ftfull", "lewm_coop2_joint")
COOP_HELDOUT = "multipusht_2a_coop_heldout.h5"  # collect.py seed=1 num_episodes=100: the readouts' fit / test data
VMAS_MODELS = ("wm_full_v2.pt", "wm_full_h_agg2.pt")  # config, z-scores and delta_std are inside each file
VMAS_DATA = ("train.pt", "heldout.pt", "train_agg2.pt")

# the official configs target stable-worldmodel's copy of LeWM; point them at this repo's jepa.py / module.py
TARGETS = {
    "stable_worldmodel.wm.lewm.LeWM": "jepa.JEPA",
    "stable_worldmodel.wm.lewm.module.Predictor": "module.ARPredictor",
    "stable_worldmodel.wm.lewm.module.Embedder": "module.Embedder",
    "stable_worldmodel.wm.lewm.module.MLP": "module.MLP",
}


def retarget(cfg):
    if isinstance(cfg, dict):
        return {k: TARGETS.get(v, v) if k == "_target_" else retarget(v) for k, v in cfg.items()}
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vmas-data", action="store_true", help="also the VMAS data (~300 MB)")
    p.add_argument("--coop-heldout", action="store_true", help="also the held-out coop scenes (700 MB)")
    p.add_argument("--pusht-data", action="store_true", help="also the official Push-T dataset (44 GB)")
    args = p.parse_args()
    root = Path(os.environ["STABLEWM_HOME"]).resolve()

    out = root / "checkpoints/lewm/pusht"
    out.mkdir(parents=True, exist_ok=True)
    hf_hub_download(LEWM_REPO, "weights.pt", local_dir=out)
    cfg = json.loads(Path(hf_hub_download(LEWM_REPO, "config.json", local_dir=out)).read_text())
    (out / "config.json").write_text(json.dumps(retarget(cfg), indent=2))
    print(f"[ckpt] {out}")

    for name in COOP_MODELS:
        out = root / "checkpoints" / name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("weights.pt", "config.json", "readout/cost_probe.pt"):
            hf_hub_download(COOP_REPO, f"{name}/{f}", local_dir=root / "checkpoints")
        print(f"[ckpt] {out}")

    for f in VMAS_MODELS + (VMAS_DATA if args.vmas_data else ()):
        hf_hub_download(COOP_REPO, f"vmas_pusht/{f}", local_dir=root)
        print(f"[{'ckpt' if f in VMAS_MODELS else 'data'}] {root / 'vmas_pusht' / f}")

    if args.coop_heldout:
        hf_hub_download(COOP_REPO, f"datasets/{COOP_HELDOUT}", local_dir=root)
        print(f"[data] {root / 'datasets' / COOP_HELDOUT}")

    if args.pusht_data:
        import zstandard
        archive = next(f for f in list_repo_files(LEWM_REPO, repo_type="dataset") if f.endswith(".zst"))
        path = Path(hf_hub_download(LEWM_REPO, archive, repo_type="dataset", local_dir=root / "archives"))
        (root / "datasets").mkdir(parents=True, exist_ok=True)
        with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as r, \
                open(root / "datasets" / archive.removesuffix(".zst"), "wb") as o:
            shutil.copyfileobj(r, o, length=64 << 20)
        path.unlink()
        print(f"[data] {root / 'datasets' / archive.removesuffix('.zst')}")


if __name__ == "__main__":
    main()
