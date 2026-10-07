"""Maintainers: publish the final coop2 weights to the Hugging Face repo that scripts/download.py reads.

    huggingface-cli login
    python scripts/upload_checkpoints.py <user>/<repo> --epoch 30
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(os.environ.get("STABLEWM_HOME", Path(__file__).resolve().parents[1] / "data"))
MODELS = ("lewm_coop2_ftfull", "lewm_coop2_joint")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--epoch", type=int, default=30)
    args = p.parse_args()
    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in MODELS:
            src, dst = ROOT / "checkpoints" / name, Path(tmp) / name
            dst.mkdir()
            shutil.copy(src / f"weights_epoch_{args.epoch}.pt", dst / "weights.pt")
            shutil.copy(src / "config.json", dst / "config.json")
        api.upload_folder(folder_path=tmp, repo_id=args.repo, commit_message=f"coop2 checkpoints, epoch {args.epoch}")
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
