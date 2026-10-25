#!/usr/bin/env python3
"""Install the pinned official CrazyMARL checkout in its isolated CPU environment.

The legacy JAX wheel has an upstream-reported executable-stack metadata defect
on current glibc. The repair changes only the ELF GNU_STACK execute bit and
records original and repaired hashes; it never changes system loader settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


UPSTREAM_URL = "https://github.com/IMRCLab/CrazyMARL.git"
UPSTREAM_COMMIT = "fca7b6e05f3b5e816be7c982c91d69eb92925f91"
XLA_ORIGINAL_SHA256 = "e25a01a770414bf2c70029af09416fbff00ed7811d8b78dfde744d8a57f38d2b"
XLA_REPAIRED_SHA256 = "5952c9f7bdea849a57214f4f7babaa44b5354c323bdfe4fef5a501982faca0bc"


def digest(value):return hashlib.sha256(value).hexdigest()


def repair_execstack(path, provenance):
    path=Path(path);before=path.read_bytes();before_hash=digest(before)
    if before_hash not in (XLA_ORIGINAL_SHA256,XLA_REPAIRED_SHA256):
        raise ValueError("Refusing to patch an unrecognized JAX CPU binary")
    modified=bytearray(before)
    if modified[:6]!=b"\x7fELF\x02\x01":raise ValueError("Expected the pinned ELF64 little-endian wheel")
    offset=struct.unpack_from("<Q",modified,32)[0]
    size=struct.unpack_from("<H",modified,54)[0]
    count=struct.unpack_from("<H",modified,56)[0]
    entries=[]
    for index in range(count):
        start=offset+index*size
        kind,flags=struct.unpack_from("<II",modified,start)
        if kind==0x6474e551:
            entries.append(dict(index=index,old_flags=flags,new_flags=flags&~1))
            struct.pack_into("<I",modified,start+4,flags&~1)
    if len(entries)!=1 or digest(modified)!=XLA_REPAIRED_SHA256:
        raise AssertionError("ELF program-header repair differs from the validated binary")
    if modified!=before:
        # Atomic replacement breaks any uv cache hard links before mutation.
        temporary=path.with_name(path.name+".noexecstack.tmp")
        if temporary.exists():raise FileExistsError(temporary)
        temporary.write_bytes(modified);temporary.chmod(path.stat().st_mode);temporary.replace(path)
    record=dict(path=str(path),jaxlib_version="0.5.1",original_sha256=XLA_ORIGINAL_SHA256,
        repaired_sha256=XLA_REPAIRED_SHA256,found_sha256=before_hash,program_headers=entries,
        repair="Clear only PT_GNU_STACK PF_X; no system-wide loader change",
        upstream_issue="https://github.com/jax-ml/jax/issues/26781")
    provenance=Path(provenance);provenance.parent.mkdir(parents=True,exist_ok=True)
    provenance.write_text(json.dumps(record,indent=2)+"\n")
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout",type=Path,default=Path("data/external/CrazyMARL"))
    parser.add_argument("--venv",type=Path,default=Path("data/external/.venv-crazymarl"))
    parser.add_argument("--python",default=".venv/bin/python")
    parser.add_argument("--repair-only",action="store_true")
    parser.add_argument("--cuda",action="store_true",help="Install pinned GPU extras; choose a separate --venv")
    parser.add_argument("--provenance",type=Path,default=Path("data/uav_transport_external/crazymarl_runtime_compat.json"))
    args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    checkout=(project/args.checkout).resolve();venv=(project/args.venv).resolve()
    if not args.repair_only:
        if not checkout.exists():
            subprocess.run(["git","clone","--depth","1","--filter=blob:none","--sparse",UPSTREAM_URL,str(checkout)],check=True)
            subprocess.run(["git","-C",str(checkout),"fetch","--depth","1","origin",UPSTREAM_COMMIT],check=True)
            subprocess.run(["git","-C",str(checkout),"checkout","--detach",UPSTREAM_COMMIT],check=True)
            subprocess.run(["git","-C",str(checkout),"sparse-checkout","set","crazymarl","jaxmarl","baselines","envs","tests"],check=True)
        head=subprocess.check_output(["git","-C",str(checkout),"rev-parse","HEAD"],text=True).strip()
        if head!=UPSTREAM_COMMIT:raise ValueError("Expected the pinned CrazyMARL commit; preserve other checkouts separately")
        dirty=subprocess.check_output(["git","-C",str(checkout),"status","--porcelain"],text=True).strip()
        if dirty:raise ValueError("Preserve the existing changed upstream checkout: "+dirty)
        if not venv.exists():
            requested=project/args.python
            python_spec=str(requested) if requested.exists() else args.python
            subprocess.run(["uv","venv","--python",python_spec,str(venv)],check=True)
        python=venv/"bin/python"
        subprocess.run(["uv","pip","install","--python",str(python),"-r",str(project/("requirements-crazymarl-gpu.txt" if args.cuda else "requirements-crazymarl.txt"))],check=True)
        subprocess.run(["uv","pip","install","--python",str(python),"--no-deps","-e",str(checkout)],check=True)
    candidates=list(venv.glob("lib/python*/site-packages/jaxlib/xla_extension.so"))
    if len(candidates)!=1:raise ValueError("Expected exactly one installed pinned JAX CPU extension")
    record=repair_execstack(candidates[0],project/args.provenance)
    print(json.dumps(record),flush=True)


if __name__=="__main__":main()
