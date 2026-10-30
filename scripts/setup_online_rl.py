#!/usr/bin/env python3
"""Install the pinned official TD-MPC2 or DreamerV3 checkout and its own venv (uav_arena/README.md).

    python3 scripts/setup_online_rl.py tdmpc2
    python3 scripts/setup_online_rl.py dreamerv3

Each method gets data/external/<method> at its pinned commit, left unmodified, and
data/external/.venv-<method> (Python 3.11) from requirements-<method>.txt. TD-MPC2's torch comes
first from PyTorch's cu128 index. jaxlib 0.5.1 requests an executable stack, which current glibc
refuses to load (jax-ml/jax#26781); as scripts/setup_crazymarl.py does, the script clears only the
ELF GNU_STACK execute bit of the recognized binary and records both hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

METHODS = dict(
    tdmpc2=dict(url="https://github.com/nicklashansen/tdmpc2",
                commit="332f19e4aba4bffe2dc736f45ea7f7bdc585903a",
                torch=("https://download.pytorch.org/whl/cu128", "torch==2.7.1")),
    dreamerv3=dict(url="https://github.com/danijar/dreamerv3",
                   commit="e01491fad6434b2245a3b8ca201dd7faedcc458c",
                   repair=dict(file="jaxlib/xla_extension.so", version="jaxlib 0.5.1 (cp311)",
                               original="0d0cf2dfb6f02a2d9d3bc71304068936315f463776f09947a2fa299f563afc91",
                               repaired="7bb3b5b305776e073b3b22a9ac4edac9020a97372e3f44d440051a7168e694a3")))
PT_GNU_STACK, PF_X = 0x6474E551, 0x1


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def clear_execstack(path, original, repaired):
    """Clear PF_X on PT_GNU_STACK of the recognized binary; returns what changed."""
    path = Path(path)
    before = path.read_bytes()
    if sha256(before) not in (original, repaired):
        raise ValueError(f"Refusing to patch an unrecognized binary: {path}")
    data = bytearray(before)
    offset, = struct.unpack_from("<Q", data, 0x20)
    size, count = struct.unpack_from("<HH", data, 0x36)
    entries = []
    for index in range(count):
        kind, flags = struct.unpack_from("<II", data, offset+index*size)
        if kind == PT_GNU_STACK:
            entries.append(dict(index=index, old_flags=flags, new_flags=flags & ~PF_X))
            struct.pack_into("<I", data, offset+index*size+4, flags & ~PF_X)
    if len(entries) != 1 or sha256(data) != repaired:
        raise AssertionError("The repair differs from the validated binary")
    if data != before:                       # replace atomically: never write through uv's cache hard links
        temporary = path.with_name(path.name+".noexecstack.tmp")
        temporary.write_bytes(data)
        temporary.chmod(path.stat().st_mode)
        temporary.replace(path)
    return dict(path=str(path), found_sha256=sha256(before), original_sha256=original, repaired_sha256=repaired,
                program_headers=entries, repair="Clear only PT_GNU_STACK PF_X; no system-wide loader change",
                upstream_issue="https://github.com/jax-ml/jax/issues/26781")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=tuple(METHODS))
    parser.add_argument("--checkout", type=Path, help="default data/external/<method>")
    parser.add_argument("--venv", type=Path, help="default data/external/.venv-<method>")
    parser.add_argument("--python", default="3.11")
    args = parser.parse_args()
    project, method = Path(__file__).resolve().parents[1], METHODS[args.method]
    checkout = (project/(args.checkout or Path("data/external")/args.method)).resolve()
    venv = (project/(args.venv or Path("data/external")/f".venv-{args.method}")).resolve()
    if not checkout.exists():
        subprocess.run(["git", "clone", method["url"], str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", method["commit"]], check=True)
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != method["commit"]:
        raise ValueError(f"{checkout} is at {head}, not the pinned commit; keep other checkouts elsewhere")
    if subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError(f"Keep the official source unchanged: {checkout}")
    if not venv.exists():
        subprocess.run(["uv", "venv", "--python", args.python, str(venv)], check=True)
    python = str(venv/"bin"/"python")
    if "torch" in method:
        index, torch = method["torch"]
        subprocess.run(["uv", "pip", "install", "--python", python, "--index-url", index, torch], check=True)
    subprocess.run(["uv", "pip", "install", "--python", python, "-r", str(project/f"requirements-{args.method}.txt")],
                   check=True)
    record = dict(method=args.method, url=method["url"], commit=head, checkout=str(checkout), venv=str(venv))
    if "repair" in method:
        repair = method["repair"]
        found = sorted(venv.glob(f"lib/python*/site-packages/{repair['file']}"))
        if len(found) != 1:
            raise ValueError(f"Expected one {repair['file']} in {venv}")
        record["repair"] = dict(clear_execstack(found[0], repair["original"], repair["repaired"]), version=repair["version"])
    (venv/"setup_online_rl.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
