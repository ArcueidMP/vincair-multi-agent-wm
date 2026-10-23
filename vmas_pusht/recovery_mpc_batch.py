"""Parallel, matched-batch execution of the frozen recovery MPC protocol.

Only independent source cases are partitioned. Both libraries receive identical
case partitions, snapshots and budgets; no candidate plans or physics steps are
removed. Every worker calls the unchanged recovery_mpc.run_mpc implementation.
The original full-batch baseline is retained as a batching sensitivity control.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from contextlib import redirect_stdout
import json
import multiprocessing
from pathlib import Path
import time

import torch

from vmas_pusht.recovery_mpc import (handoff_record, select_record, run_mpc,
    protocol, sha, summary)
from vmas_pusht.boundary import _plain


def worker(source_path, output, library, batch, indices, substeps):
    torch.set_num_threads(1)
    source = torch.load(source_path, map_location="cpu")
    record = select_record(handoff_record(source), indices)
    prefix = Path(f"{output}_{library}_batch{batch}")
    with prefix.with_suffix(".log").open("w") as stream, redirect_stdout(stream):
        result = run_mpc(record, library, substeps=substeps)
    raw = prefix.with_suffix(".pt")
    torch.save(result, raw)
    report = dict(library=library, batch=batch, global_indices=indices,
        raw_path=str(raw), raw_sha256=sha(raw), rows=result["rows"],
        summary=summary(result["rows"]), elapsed_seconds=result["elapsed_seconds"])
    path = prefix.with_suffix(".json")
    path.write_text(json.dumps(_plain(report), indent=2, allow_nan=False)+"\n")
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/novelty/stable_composition_selected_matched.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/novelty/recovery_mpc_parallel"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=54)
    parser.add_argument("--substeps", type=int, default=160)
    args = parser.parse_args()
    if min(args.workers, args.batch_size) < 1:
        raise ValueError("Positive worker and batch counts required")
    torch.set_num_threads(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha(args.source)
    source = torch.load(args.source, map_location="cpu")
    record = handoff_record(source)
    n = len(record["rows"])
    partitions = [list(range(i, min(i+args.batch_size, n))) for i in range(0, n, args.batch_size)]
    code_paths = [Path(__file__)] + [Path("vmas_pusht")/(name+".py") for name in
        ("recovery_mpc", "recovery_primitives", "recovery_attribution", "physics_profiles",
         "snapshot", "stable_composition", "abstraction_convergence", "physical_abstraction",
         "abstraction_transfer", "scenario", "boundary")]
    models = [Path("data/novelty")/(name+".pt") for name in
        ("stable_local_original_models", "stable_local_selected_models")]
    code_hashes = {str(p): sha(p) for p in code_paths}
    model_hashes = {str(p): sha(p) for p in models}
    declaration = dict(protocol=protocol(), source_path=str(args.source), source_sha256=source_digest,
        source_hashes=code_hashes, frozen_models_sha256=model_hashes,
        case_count=n, batch_size=args.batch_size, workers=args.workers, substeps=args.substeps,
        case_partitions=partitions,
        batching_note="Same partition for both libraries. Full states unchanged; float batch effects audited separately. Previous full-batch baseline preserved.")
    registration = args.output.with_name(args.output.name+"_registration.json")
    if registration.exists():
        raise FileExistsError("Choose a fresh output prefix to preserve the registered run")
    registration.write_text(json.dumps(declaration, indent=2)+"\n")
    started = time.perf_counter()
    completed = {name: {} for name in ("bank13", "expanded")}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, args.source, args.output, library, batch, indices, args.substeps):
                   (library, batch) for library in completed for batch, indices in enumerate(partitions)}
        for future in as_completed(pending):
            library, batch = pending[future]
            path = Path(future.result())
            report = json.loads(path.read_text())
            completed[library][batch] = dict(report_path=str(path), report_sha256=sha(path), **report)
            print(json.dumps(dict(library=library, batch=batch,
                rows=len(report["rows"]), ordered=sum(r["ordered_goal_success"] for r in report["rows"]),
                continuation=sum(r["continuation_goal_success"] for r in report["rows"]),
                elapsed_seconds=time.perf_counter()-started)), flush=True)
            if len(completed[library]) == len(partitions):
                batches = [completed[library][i] for i in range(len(partitions))]
                rows = [row for b in batches for row in b["rows"]]
                if len(rows) != n or len({(r["source_group"], r["source_row"]) for r in rows}) != n:
                    raise AssertionError("Incomplete or duplicated case coverage")
                aggregate = dict(declaration, registration_path=str(registration), registration_sha256=sha(registration),
                    library=library, rows=rows, summary=summary(rows),
                    batches=[{k: b[k] for k in ("batch", "global_indices", "raw_path", "raw_sha256", "report_path", "report_sha256")} for b in batches],
                    wall_seconds=time.perf_counter()-started)
                path = Path(f"{args.output}_{library}.json")
                path.write_text(json.dumps(aggregate, indent=2)+"\n")
    for path, digest in {str(args.source): source_digest, **code_hashes, **model_hashes}.items():
        if sha(path) != digest:
            raise AssertionError("Frozen source/model changed: "+path)
    print(json.dumps(dict(output=str(args.output), wall_seconds=time.perf_counter()-started,
        verified_input_and_model_hashes=True)), flush=True)


if __name__ == "__main__":
    main()
