"""Maintainers: publish the released weights to the Hugging Face repo that scripts/download.py reads.

    hf auth login
    python scripts/upload_checkpoints.py fuisloy/vincair-multi-agent-wm --epoch 30

Repo layout (one commit):
    README.md                                    scripts/model_card.md
    lewm_coop2_{ftfull,joint}/weights.pt         $STABLEWM_HOME/checkpoints/<name>/weights_epoch_<epoch>.pt
    lewm_coop2_{ftfull,joint}/config.json
    lewm_coop2_{ftfull,joint}/readout/cost_probe.pt  $STABLEWM_HOME/checkpoints/<name>/readout/ (scripts/fit_readout.py)
    datasets/multipusht_2a_coop_heldout.h5       held-out coop scenes (--no-coop-heldout to skip)
    vmas_pusht/wm_full_v2.pt, wm_full_h_agg2.pt  $STABLEWM_HOME/vmas_pusht/ (config and z-scores inside each file)
    vmas_pusht/{train,heldout,train_agg2}.pt     VMAS data (--no-vmas-data to skip)
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(os.environ.get("STABLEWM_HOME", Path(__file__).resolve().parents[1] / "data"))
MODELS = ("lewm_coop2_ftfull", "lewm_coop2_joint")
VMAS_MODELS = ("wm_full_v2.pt", "wm_full_h_agg2.pt")
VMAS_DATA = ("train.pt", "heldout.pt", "train_agg2.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--message", default="coop2 and VMAS checkpoints")
    p.add_argument("--no-vmas-data", action="store_true")
    p.add_argument("--no-coop-heldout", action="store_true")
    p.add_argument("--only", nargs="+", help="upload only these repo paths (e.g. a new readout)")
    args = p.parse_args()
    files = {"README.md": Path(__file__).with_name("model_card.md")}
    for name in MODELS:
        src = ROOT / "checkpoints" / name
        files[f"{name}/weights.pt"] = src / f"weights_epoch_{args.epoch}.pt"
        files[f"{name}/config.json"] = src / "config.json"
        files[f"{name}/readout/cost_probe.pt"] = src / "readout" / "cost_probe.pt"
    if not args.no_coop_heldout:
        files["datasets/multipusht_2a_coop_heldout.h5"] = ROOT / "datasets" / "multipusht_2a_coop_heldout.h5"
    for f in VMAS_MODELS + (() if args.no_vmas_data else VMAS_DATA):
        files[f"vmas_pusht/{f}"] = ROOT / "vmas_pusht" / f
    if args.only:
        files = {k: v for k, v in files.items() if k in args.only}
    missing = [str(v) for v in files.values() if not v.exists()]
    assert not missing, f"missing: {missing}"

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    api.create_commit(args.repo, operations=ops, commit_message=args.message)
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
