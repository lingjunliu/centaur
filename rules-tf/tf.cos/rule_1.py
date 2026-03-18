import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 1: tf.cos input tensor constraints
"""
1. x must be a tensor
2. x must have numeric dtype (float or complex)
3. Not allowed: bool, string
4. Rank must be valid
5. All dimensions must be positive
"""

rule_1 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # Valid rank
                v["x_ndim"] >= 0,
                v["x_ndim"] <= MAX_N_DIM,

                # All dimensions positive
                And([
                    Implies(i < v["x_ndim"],
                            And(
                                Select(v["x_shape"], i) > 0,
                                Select(v["x_shape"], i) <= MAX_SZ_DIM
                            ))
                    for i in range(MAX_N_DIM)
                ]),

                # Numeric dtype only (exclude bool and string)
                And([
                    v["x_dtype"] != list_of_available_dtypes.index(dt)
                    for dt in list_of_string_values_tf
                ])
            )
        ) if n else
        And(
            v["x_ndim"] >= 0,
            v["x_ndim"] <= MAX_N_DIM,

            And([
                Implies(i < v["x_ndim"],
                        And(
                            Select(v["x_shape"], i) > 0,
                            Select(v["x_shape"], i) <= MAX_SZ_DIM
                        ))
                for i in range(MAX_N_DIM)
            ]),

            And([
                v["x_dtype"] != list_of_available_dtypes.index(dt)
                for dt in list_of_string_values_tf
            ])
        )
    )
)

def rule_1_func(arg1, solver=None, neg=False):
    x = next(iter(arg1.values()))

    # -------------------------
    # Invariant learning phase
    # -------------------------
    if not solver:
        if not isinstance(x, np.ndarray):
            return False

        solver = Solver()

        # Variable declarations
        x_ndim = Int('x_ndim')
        x_shape = Array('x_shape', IntSort(), IntSort())
        x_dtype = Int('x_dtype')

        # Assign concrete values
        solver.add(x_ndim == x.ndim)

        for i in range(x.ndim):
            x_shape = Store(x_shape, i, x.shape[i])

        solver.add(x_dtype == list_of_available_dtypes.index(x.dtype))

        # Apply rule
        rule_1(solver, {
            "x_ndim": x_ndim,
            "x_shape": x_shape,
            "x_dtype": x_dtype
        })

        return solver.check() == sat

    # -------------------------
    # Fuzz generation phase
    # -------------------------
    else:
        rule_1(
            solver,
            {
                "x_ndim": x["ndim"],
                "x_shape": x["shape"],
                "x_dtype": x["dtype"]
            },
            neg
        )