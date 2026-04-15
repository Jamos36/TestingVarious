"""
coordinator.py — rank 0 only.

Sends jobs to workers, collects results, writes output.
Calls NO simulation code directly — that all lives in pipeline.py.

WHEN to use which pattern:

  Dynamic queue (below):
    → jobs vary in runtime (some take 10s, some take 10min)
    → you want load balancing — fast workers get more jobs
    → this is almost always the right choice for parameter sweeps

  Scatter (see comments at bottom):
    → all jobs take identical time
    → you want the least communication code
    → rank 0 also does computation (no dedicated coordinator)
"""

from mpi4py import MPI

STOP_TAG = 99


def run_coordinator(comm, size: int, config: dict):
    from main import build_jobs, save_result

    jobs     = build_jobs(config)
    n_jobs   = len(jobs)
    n_workers = size - 1
    results  = {}
    errors   = []

    # Seed each worker with its first job
    job_idx = 0
    active  = 0
    for w in range(1, min(n_workers + 1, n_jobs + 1)):
        comm.send(jobs[job_idx], dest=w, tag=job_idx)
        job_idx += 1
        active += 1

    # Feed loop — as each result comes in, dispatch the next job
    while active > 0:
        status = MPI.Status()
        result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        worker = status.Get_source()
        active -= 1

        # Handle errors without aborting
        if "error" in result:
            print(f"[rank 0] Job {result['sim_id']} FAILED: {result['error']}", flush=True)
            errors.append(result)
        else:
            results[result["sim_id"]] = result
            save_result(result, config)   # write as they arrive — don't wait for all

        # Send next job to the now-free worker
        if job_idx < n_jobs:
            comm.send(jobs[job_idx], dest=worker, tag=job_idx)
            job_idx += 1
            active += 1

    # Shutdown all workers
    for w in range(1, n_workers + 1):
        comm.send(None, dest=w, tag=STOP_TAG)

    print(f"[rank 0] Done. {len(results)} succeeded, {len(errors)} failed.", flush=True)
    if errors:
        import json
        with open("errors.json", "w") as f:
            json.dump(errors, f, indent=2)


# ------------------------------------------------------------------ #
# SCATTER VARIANT — only use when all jobs are equal cost             #
# ------------------------------------------------------------------ #
# If you switch to this, remove run_coordinator above and use this.
# Note: rank 0 also processes jobs here (no dedicated coordinator).
# This means if rank 0 is slow (e.g. it's doing file I/O too),
# it becomes the bottleneck for everyone.
#
# def run_coordinator_scatter(comm, size, config):
#     from main import build_jobs, save_result
#     from pipeline import run_full_pipeline
#
#     jobs = build_jobs(config)
#     while len(jobs) % size != 0:
#         jobs.append(None)
#
#     chunk = len(jobs) // size
#     chunks = [jobs[i*chunk:(i+1)*chunk] for i in range(size)]
#
#     my_jobs = comm.scatter(chunks, root=0)
#     my_results = [run_full_pipeline(j, config) for j in my_jobs if j]
#     all_results = comm.gather(my_results, root=0)
#
#     if rank == 0:
#         for r in (r for sub in all_results for r in sub):
#             save_result(r, config)
