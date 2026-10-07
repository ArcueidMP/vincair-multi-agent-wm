#!/bin/bash
# Install the environment and download the checkpoints.
#   bash scripts/install.sh               # data under ./data
#   bash scripts/install.sh /big/disk     # data under /big/disk, linked as ./data
set -e
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
if [ -n "$1" ]; then
    mkdir -p "$1"
    [ -e data ] || ln -s "$(realpath "$1")" data
else
    mkdir -p data
fi
uv sync                                   # .venv from uv.lock (PyTorch CUDA 12.8 wheels)
uv run python scripts/download.py         # official LeWM + coop2 checkpoints (~210 MB)
uv run python -c "from world import MultiPushT; e = MultiPushT(agent_force=1e5, solver_iterations=50); e.reset(seed=0); print('MultiPushT ok')"
