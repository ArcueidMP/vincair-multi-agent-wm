"""Pinned official checkouts of external methods (TD-MPC2, DreamerV3): provenance and imports.

The official code runs unchanged: a launcher checks that a checkout sits at its pinned commit with
an unmodified tracked tree, then imports it and registers the arena task the way the code registers
its own tasks.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
import sys


def _git(checkout, *args):
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def provenance(checkout, url, commit):
    """The checkout's URL and commit, or an error if it is not at `commit` or its tracked files changed."""
    checkout = Path(checkout).resolve()
    head = _git(checkout, "rev-parse", "HEAD")
    if head != commit:
        raise ValueError(f"{checkout} is at {head}, not the pinned commit {commit}")
    changed = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    if changed:
        raise ValueError(f"Keep the official source unchanged: {changed}")
    return dict(url=url, commit=head, checkout=str(checkout), tracked_source_unchanged=True)


@contextmanager
def importable(path):
    """Put `path` first on `sys.path` while the context lasts."""
    saved = sys.path.copy()
    sys.path.insert(0, str(Path(path).resolve()))
    try:
        yield
    finally:
        sys.path[:] = saved


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    """`value` with paths, tuples and NumPy scalars made JSON-friendly."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    return value
