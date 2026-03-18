import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Valid input for tf.linalg.tensor_diag

rule_1 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["diag_ndim"] >= 1,
                And([
                    Implies(
                        i < v["diag_ndim"],
                        Select(v["diag_shape"], i) > 0
                    )
                    for i in range(MAX_N_DIM)
                ])
            )
        )
    ) if n else
    s.add(
        And(
            v["diag_ndim"] >= 1,
            And([
                Implies(
                    i < v["diag_ndim"],
                    Select(v["diag_shape"], i) > 0
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_1_func(arg1, solver=None, neg=False):

    diagonal = next(iter(arg1.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(diagonal, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        diag_ndim = Int("diag_ndim")
        diag_shape = Array("diag_shape", IntSort(), IntSort())

        # Concrete assignments
        solver.add(diag_ndim == diagonal.ndim)

        for i in range(diagonal.ndim):
            diag_shape = Store(diag_shape, i, diagonal.shape[i])

        # Apply constraint
        rule_1(
            solver,
            {
                "diag_ndim": diag_ndim,
                "diag_shape": diag_shape,
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:

        rule_1(
            solver,
            {
                "diag_ndim": diagonal["ndim"],
                "diag_shape": diagonal["shape"],
            },
            neg
        )