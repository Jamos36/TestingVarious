"""
solver.py — The computationally expensive part.

Nothing here knows about MPI. This is just science/math code.
Each worker runs its own independent copy of this.

If your solver itself is parallelisable (e.g. scipy, numba, C extension),
it can use threads or multiprocessing internally — that is fine and
complementary to MPI. MPI handles inter-node parallelism;
threading handles intra-node parallelism.
"""

import numpy as np
from scipy.integrate import solve_ivp   # example — use whatever you actually use


def run_solver(initial_state: np.ndarray, job: dict, config: dict) -> np.ndarray:
    """
    Your tough calculation. Receives the prepared state, returns the solution.
    No MPI. No special changes from what you already have.
    """
    alpha = job["alpha"]
    beta  = job["beta"]
    t_end = config.get("t_end", 100.0)

    def odes(t, y):
        return [-alpha * y[0] + beta * y[1],
                 alpha * y[0] - beta * y[1]]

    sol = solve_ivp(odes, [0, t_end], initial_state[:2],
                    method="RK45", dense_output=False,
                    rtol=1e-8, atol=1e-10)

    return sol.y[:, -1]   # return final state
