"""Verify this frozen archive, optionally checking referenced external raw data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def within(root, path):
    value = (root/path).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("Archive reference leaves its declared root: "+path)
    return value


def verify(external_root=None):
    manifest_path = ARCHIVE/"manifest.json"
    expected = (ARCHIVE/"manifest.sha256").read_text().split()[0]
    if digest(manifest_path) != expected:
        raise AssertionError("Archive manifest differs from its detached checksum")
    manifest = json.loads(manifest_path.read_text())
    references = [(x["archive_path"], x["sha256"]) for x in manifest["captured_files"]]
    references += [(x["path"], x["sha256"]) for x in manifest["generated_files"]]
    for path, expected in references:
        if digest(within(ARCHIVE, path)) != expected:
            raise AssertionError("Archived file changed: "+path)
    external = []
    if external_root is not None:
        index = json.loads((ARCHIVE/"external_artifacts.json").read_text())
        external = index["artifacts"]
        root = Path(external_root).resolve()
        for entry in external:
            if digest(within(root, entry["path"])) != entry["sha256"]:
                raise AssertionError("Referenced raw artifact changed: "+entry["path"])
    return dict(status="verified", captured_files=len(manifest["captured_files"]),
        generated_files=len(manifest["generated_files"]), external_raw_files_verified=len(external),
        git_head_at_capture=manifest["git_head"], baseline_family=manifest["baseline_family"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path,
        help="Original or relocated project root carrying data/uav_transport; files are read only")
    args = parser.parse_args()
    print(json.dumps(verify(args.external_root), indent=2))
