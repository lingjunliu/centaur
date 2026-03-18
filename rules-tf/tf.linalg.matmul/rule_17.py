import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: Shape compatibility for tf.linalg.matmul (default flags)
"""
1. rank(a) >= 2
2. rank(b) >= 2
3. rank(a) == rank(b) (TensorFlow requires equal ranks for batch matmul)
4. For every batch dimension i < rank-2: a.shape[i] == b.shape[i]
5. Inner matrix compatibility: a.shape[rank-1] of A must equal b.shape[rank-2]
"""

rule_17 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank >= 2
                v["a_ndim"] >= 2,
                v["b_ndim"] >= 2,

                # ranks equal
                v["a_ndim"] == v["b_ndim"],

                # inner dimension match
                Select(v["a_shape"], v["a_ndim"] - 1) ==
                Select(v["b_shape"], v["b_ndim"] - 2),

                # batch dimensions equal
                And([
                    Select(v["a_shape"], i) ==
                    Select(v["b_shape"], i)
                    for i in range(MAX_N_DIM - 2)
                ])
            )
        )
    ) if n else
    s.add(
        And(
            v["a_ndim"] >= 2,
            v["b_ndim"] >= 2,
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

def rule_17_func(arg1, arg2, solver=None, neg=False):

    a = next(iter(arg1.values()))
    b = next(iter(arg2.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return False

        solver = Solver()

        # Declare variables
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
        rule_17(
            solver,
            {
                "a_ndim": a_ndim,
                "b_ndim": b_ndim,
                "a_shape": a_shape,
                "b_shape": b_shape,
            },
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz input generation phase
    # -----------------------------
    else:

        rule_17(
            solver,
            {
                "a_ndim": a["ndim"],
                "b_ndim": b["ndim"],
                "a_shape": a["shape"],
                "b_shape": b["shape"],
            },
            neg
        )