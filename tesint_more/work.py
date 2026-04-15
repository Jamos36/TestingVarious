"""
worker.py — MPI coordination shell for ranks 1..N-1.

This file's entire job is:
  1. Receive a job dict from rank 0  (1 MPI call)
  2. Call run_full_pipeline()         (0 MPI calls — pure Python)
  3. Send the result back to rank 0  (1 MPI call)
  4. Repeat until told to stop

That is it. Two MPI calls per job. Everything else is your code.
"""

from mpi4py import MPI
from pipeline import run_full_pipeline

STOP_TAG = 99


def run_worker(comm, rank: int, config: dict):
    while True:
        # ---- MPI TOUCH POINT 1: receive job assignment ----
        status = MPI.Status()
        job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == STOP_TAG:
            # No more work. Exit cleanly.
            break

        # ---- ENTIRE PIPELINE: zero MPI, all your existing code ----
        try:
            result = run_full_pipeline(job, config)
        except Exception as e:
            # Don't crash the whole run — report back to rank 0
            result = {
                "sim_id": job.get("sim_id", "unknown"),
                "error":  str(e),
                "status": "failed"
            }

        # ---- MPI TOUCH POINT 2: send result back ----
        comm.send(result, dest=0, tag=tag)
