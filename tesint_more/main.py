#!/usr/bin/env python3
"""
main.py — entry point run by EVERY rank.

Think of MPI as a post office. This file is the post office's sorting room.
Your actual simulation code (simulation.py, solver.py, etc.) never needs to
know the post office exists.
"""

import argparse
import yaml

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--serial", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)

    # ------------------------------------------------------------------ #
    # SERIAL PATH — import nothing MPI-related. Clean fallback.           #
    # Good for: one large run, debugging, no cluster available.           #
    # ------------------------------------------------------------------ #
    if args.serial:
        from pipeline import run_full_pipeline   # your existing pipeline, untouched
        jobs = build_jobs(config)
        for job in jobs:
            result = run_full_pipeline(job, config)
            save_result(result, config)
        return

    # ------------------------------------------------------------------ #
    # MPI PATH                                                            #
    # ------------------------------------------------------------------ #
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size == 1:
        # Ran without srun/mpirun by accident — degrade gracefully
        print("WARNING: only 1 MPI process. Running serial.", flush=True)
        from pipeline import run_full_pipeline
        jobs = build_jobs(config)
        for job in jobs:
            result = run_full_pipeline(job, config)
            save_result(result, config)
        return

    if rank == 0:
        from coordinator import run_coordinator
        run_coordinator(comm, size, config)
    else:
        from worker import run_worker
        run_worker(comm, rank, config)

def build_jobs(config: dict) -> list:
    """Build the job list. Used by serial path and coordinator."""
    return [
        {"sim_id": p["id"], **p}
        for p in config["parameter_sweep"]
    ]

def save_result(result, config):
    import json, os
    out = config.get("output_dir", "output")
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/{result['sim_id']}.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()
