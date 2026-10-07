"""Download checkpoints into $STABLEWM_HOME/checkpoints (default ./data/checkpoints).

    python scripts/download.py                 # official LeWM Push-T checkpoint + the two coop2 models (~210 MB)
    python scripts/download.py --pusht-data    # also the official Push-T dataset (44 GB, only to re-export assets/)

Layout:
    checkpoints/lewm/pusht/{weights.pt,config.json}          official single-agent LeWM (quentinll/lewm-pusht)
    checkpoints/lewm_coop2_ftfull/{weights.pt,config.json}   fine-tuned on coop data, own action
    checkpoints/lewm_coop2_joint/{weights.pt,config.json}    fine-tuned on coop data, action = [self, partner]
"""
import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("STABLEWM_HOME", str(Path(__file__).resolve().parents[1] / "data"))

from huggingface_hub import hf_hub_download, list_repo_files

LEWM_REPO = "quentinll/lewm-pusht"
COOP_REPO = os.environ.get("COOP_REPO", "TODO/multipusht-lewm")  # Hugging Face model repo with the coop2 checkpoints
COOP_MODELS = ("lewm_coop2_ftfull", "lewm_coop2_joint")

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
    p.add_argument("--pusht-data", action="store_true", help="also the official Push-T dataset (44 GB)")
    args = p.parse_args()
    root = Path(os.environ["STABLEWM_HOME"]).resolve()

    out = root / "checkpoints/lewm/pusht"
    out.mkdir(parents=True, exist_ok=True)
    hf_hub_download(LEWM_REPO, "weights.pt", local_dir=out)
    cfg = json.loads(Path(hf_hub_download(LEWM_REPO, "config.json", local_dir=out)).read_text())
    (out / "config.json").write_text(json.dumps(retarget(cfg), indent=2))
    print(f"[ckpt] {out}")

    if COOP_REPO.startswith("TODO"):
        print("[ckpt] coop2 models skipped: set COOP_REPO (in scripts/download.py or the environment) to their Hugging Face repo")
    for name in () if COOP_REPO.startswith("TODO") else COOP_MODELS:
        out = root / "checkpoints" / name
        out.mkdir(parents=True, exist_ok=True)
        for f in ("weights.pt", "config.json"):
            hf_hub_download(COOP_REPO, f"{name}/{f}", local_dir=root / "checkpoints")
        print(f"[ckpt] {out}")

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
