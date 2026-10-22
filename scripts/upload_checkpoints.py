"""Maintainers: publish the released weights to the Hugging Face repo that scripts/download.py reads.

    hf auth login
    python scripts/upload_checkpoints.py fuisloy/vincair-multi-agent-wm --epoch 30

Repo layout (one commit):
    README.md                                    scripts/model_card.md
    lewm_coop2_{ftfull,joint}/weights.pt         $STABLEWM_HOME/checkpoints/<name>/weights_epoch_<epoch>.pt
    lewm_coop2_{ftfull,joint}/config.json
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
    p.add_argument("--no-vmas-data", action="store_true")
    args = p.parse_args()
    files = {"README.md": Path(__file__).with_name("model_card.md")}
    for name in MODELS:
        src = ROOT / "checkpoints" / name
        files[f"{name}/weights.pt"] = src / f"weights_epoch_{args.epoch}.pt"
        files[f"{name}/config.json"] = src / "config.json"
    for f in VMAS_MODELS + (() if args.no_vmas_data else VMAS_DATA):
        files[f"vmas_pusht/{f}"] = ROOT / "vmas_pusht" / f
    missing = [str(v) for v in files.values() if not v.exists()]
    assert not missing, f"missing: {missing}"

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in files.items()]
    api.create_commit(args.repo, operations=ops, commit_message=f"coop2 (epoch {args.epoch}) and VMAS checkpoints")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
