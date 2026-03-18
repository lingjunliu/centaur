import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: x must be broadcastable with valid y

rule_5 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # rank valid
                v["x_ndim"] >= 0,

                # dimensions positive
                And([
                    Implies(
                        i < v["x_ndim"],
                        Select(v["x_shape"], i) > 0
                    )
                    for i in range(MAX_N_DIM)
                ]),

                # Broadcasting compatibility (right-aligned)
                And([
                    Or(
                        # equal dims
                        Select(v["x_shape"], v["x_ndim"] - 1 - i) ==
                        Select(v["y_shape"], v["y_ndim"] - 1 - i),

                        # x dim is 1
                        Select(v["x_shape"], v["x_ndim"] - 1 - i) == 1,

                        # y dim is 1
                        Select(v["y_shape"], v["y_ndim"] - 1 - i) == 1,

                        # x missing leading dim → implicit 1
                        i >= v["x_ndim"]
                    )
                    for i in range(MAX_N_DIM)
                ])
            )
        ) if n else
        And(
            v["x_ndim"] >= 0,

            And([
                Implies(
                    i < v["x_ndim"],
                    Select(v["x_shape"], i) > 0
                )
                for i in range(MAX_N_DIM)
            ]),

            And([
                Or(
                    Select(v["x_shape"], v["x_ndim"] - 1 - i) ==
                    Select(v["y_shape"], v["y_ndim"] - 1 - i),

                    Select(v["x_shape"], v["x_ndim"] - 1 - i) == 1,
                    Select(v["y_shape"], v["y_ndim"] - 1 - i) == 1,

                    i >= v["x_ndim"]
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_5_func(arg1, arg2, solver=None, neg=False):

    x = next(iter(arg1.values()))
    y = next(iter(arg2.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(x, np.ndarray):
            return False
        if not isinstance(y, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        x_ndim = Int('x_ndim')
        y_ndim = Int('y_ndim')

        x_shape = Array('x_shape', IntSort(), IntSort())
        y_shape = Array('y_shape', IntSort(), IntSort())

        solver.add(x_ndim == x.ndim)
        solver.add(y_ndim == y.ndim)

        for i in range(x.ndim):
            x_shape = Store(x_shape, i, x.shape[i])

        for i in range(y.ndim):
            y_shape = Store(y_shape, i, y.shape[i])

        rule_5(
            solver,
            {
                "x_ndim": x_ndim,
                "y_ndim": y_ndim,
                "x_shape": x_shape,
                "y_shape": y_shape
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:
        rule_5(
            solver,
            {
                "x_ndim": x["ndim"],
                "y_ndim": y["ndim"],
                "x_shape": x["shape"],
                "y_shape": y["shape"]
            },
            neg
        )