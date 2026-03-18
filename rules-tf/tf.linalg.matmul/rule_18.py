import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Valid shape for tensor 'a' given tensor 'b'
rule_18 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank constraints
                v["a_ndim"] >= 2,
                v["a_ndim"] == v["b_ndim"],

                # inner dimension compatibility
                Select(v["a_shape"], v["a_ndim"] - 1) ==
                Select(v["b_shape"], v["b_ndim"] - 2),

                # batch dimensions must match
                And([
                    Implies(
                        i < v["a_ndim"] - 2,
                        Select(v["a_shape"], i) ==
                        Select(v["b_shape"], i)
                    )
                    for i in range(MAX_N_DIM)
                ])
            )
        )
    ) if n else
    s.add(
        And(
            v["a_ndim"] >= 2,
            v["a_ndim"] == v["b_ndim"],
            Select(v["a_shape"], v["a_ndim"] - 1) ==
            Select(v["b_shape"], v["b_ndim"] - 2),
            And([
                Implies(
                    i < v["a_ndim"] - 2,
                    Select(v["a_shape"], i) ==
                    Select(v["b_shape"], i)
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_18_func(arg1, arg2, solver=None, neg=False):

    a = next(iter(arg1.values()))
    b = next(iter(arg2.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return False

        solver = Solver()

        # Declare symbolic variables
        a_ndim = Int("a_ndim")
        b_ndim = Int("b_ndim")

        a_shape = Array("a_shape", IntSort(), IntSort())
        b_shape = Array("b_shape", IntSort(), IntSort())

        # Assign actual values
        solver.add(a_ndim == a.ndim)
        solver.add(b_ndim == b.ndim)

        for i in range(a.ndim):
            a_shape = Store(a_shape, i, a.shape[i])

        for i in range(b.ndim):
            b_shape = Store(b_shape, i, b.shape[i])

        # Apply constraint
        rule_18(
            solver,
            {
                "a_ndim": a_ndim,
                "b_ndim": b_ndim,
                "a_shape": a_shape,
                "b_shape": b_shape,
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz input generation phase
    # -----------------------------
    else:

        rule_18(
            solver,
            {
                "a_ndim": a["ndim"],
                "b_ndim": b["ndim"],
                "a_shape": a["shape"],
                "b_shape": b["shape"],
            },
            neg
        )