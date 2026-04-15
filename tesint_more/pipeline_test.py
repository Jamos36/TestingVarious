"""
pipeline.py — The full per-job computation pipeline.

This module has NO imports from mpi4py. It does not know it is running
inside an MPI job. It just takes a job dict, runs the full pipeline,
and returns a result dict.

This is intentional. Your pandas setup, data loading, solver calls —
none of that changes from what you already have. MPI is the wrapper
around this, not the inside of it.
"""

import pandas as pd
import numpy as np

# ---- Import your solver / heavy calculator ----
from solver import run_solver   # your tough calculation lives here


def run_full_pipeline(job: dict, config: dict) -> dict:
    """
    This is what each worker calls after receiving its job assignment.
    It runs start to finish, entirely in regular Python.

    job: a dict built by the coordinator from config.yaml
         e.g. {"sim_id": "run_042", "alpha": 0.3, "beta": 1.7, ...}

    config: the full config dict (global settings, file paths, etc.)

    Returns a result dict that gets sent back to rank 0.
    """

    sim_id = job["sim_id"]
    print(f"[worker] starting {sim_id}", flush=True)

    # -------------------------------------------------------------- #
    # STAGE 1: Setup — pandas, numpy, whatever you already have       #
    # No MPI here. This is just normal Python.                        #
    # -------------------------------------------------------------- #
    df = load_input_data(job, config)
    df = preprocess(df, job)
    initial_state = build_initial_state(df, job)

    # -------------------------------------------------------------- #
    # STAGE 2: The tough calculation                                   #
    # Still no MPI here. The solver is just a function call.          #
    # It can use all CPU cores on the node via threading/numpy/scipy  #
    # if needed — MPI and threading are orthogonal.                   #
    # -------------------------------------------------------------- #
    solution = run_solver(initial_state, job, config)

    # -------------------------------------------------------------- #
    # STAGE 3: Post-process                                            #
    # Reduce the solution to whatever you need to send back.          #
    # Don't send huge arrays unless you have to — send summaries.     #
    # -------------------------------------------------------------- #
    result = postprocess(solution, job)

    print(f"[worker] finished {sim_id}", flush=True)
    return result


# ---- Your pipeline helper functions below ----
# These are exactly what you already have. Nothing changes.

def load_input_data(job: dict, config: dict) -> pd.DataFrame:
    """
    Load whatever input data this job needs.
    Each worker loads its own data independently — no coordination needed.
    If all jobs share the same input file, every worker reads it.
    That is fine for files up to ~hundreds of MB. For truly huge shared
    input data, have rank 0 Bcast a numpy array (see communications reference).
    """
    data_path = config.get("input_data", "data/input.csv")
    df = pd.read_csv(data_path)
    # filter to this job's relevant rows if needed
    return df[df["scenario"] == job.get("scenario", "default")]


def preprocess(df: pd.DataFrame, job: dict) -> pd.DataFrame:
    """Your existing preprocessing. Unchanged."""
    df = df.copy()
    df["scaled"] = df["value"] * job["alpha"]
    # ... all your pandas transformations ...
    return df


def build_initial_state(df: pd.DataFrame, job: dict) -> np.ndarray:
    """Convert dataframe to whatever your solver needs."""
    return df["scaled"].to_numpy()


def postprocess(solution, job: dict) -> dict:
    """
    Convert solver output to a serializable result dict.
    Keep this small — it gets sent over MPI back to rank 0.
    Don't return entire dataframes if you can return summary statistics instead.
    If you genuinely need to save large output per run, write it to a file
    here (each worker writes its own file) and return just the filename.
    """
    return {
        "sim_id":    job["sim_id"],
        "max_val":   float(np.max(solution)),
        "mean_val":  float(np.mean(solution)),
        "converged": bool(solution[-1] < 1e-6),
        # OR: write a file and return the path
        # "output_file": f"output/{job['sim_id']}.npy"
    }
