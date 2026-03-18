import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: x and y must be broadcastable

rule_1 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # ranks valid
                v["x_ndim"] >= 0,
                v["y_ndim"] >= 0,

                # Broadcasting condition over max dims
                And([
                    Or(
                        # equal dims
                        Select(v["x_shape"], v["x_ndim"] - 1 - i) ==
                        Select(v["y_shape"], v["y_ndim"] - 1 - i),

                        # x dim is 1
                        Select(v["x_shape"], v["x_ndim"] - 1 - i) == 1,

                        # y dim is 1
                        Select(v["y_shape"], v["y_ndim"] - 1 - i) == 1,

                        # one tensor does not have this dim (implicit 1)
                        i >= v["x_ndim"],
                        i >= v["y_ndim"]
                    )
                    for i in range(MAX_N_DIM)
                ])
            )
        ) if n else
        And(
            v["x_ndim"] >= 0,
            v["y_ndim"] >= 0,

            And([
                Or(
                    Select(v["x_shape"], v["x_ndim"] - 1 - i) ==
                    Select(v["y_shape"], v["y_ndim"] - 1 - i),

                    Select(v["x_shape"], v["x_ndim"] - 1 - i) == 1,

                    Select(v["y_shape"], v["y_ndim"] - 1 - i) == 1,

                    i >= v["x_ndim"],
                    i >= v["y_ndim"]
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_1_func(arg1, arg2, solver=None, neg=False):

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

        # Assign ranks
        solver.add(x_ndim == x.ndim)
        solver.add(y_ndim == y.ndim)

        # Assign shapes
        for i in range(x.ndim):
            x_shape = Store(x_shape, i, x.shape[i])

        for i in range(y.ndim):
            y_shape = Store(y_shape, i, y.shape[i])

        # Apply broadcasting rule
        rule_1(
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
        rule_1(
            solver,
            {
                "x_ndim": x["ndim"],
                "y_ndim": y["ndim"],
                "x_shape": x["shape"],
                "y_shape": y["shape"]
            },
            neg
        )