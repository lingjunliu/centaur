import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 2: tf.divide constraints on x (given valid y)

rule_2 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # Valid rank for x
                v["x_ndim"] >= 0,
                v["x_ndim"] <= MAX_N_DIM,

                # Positive dimensions
                And([
                    Implies(i < v["x_ndim"],
                            And(
                                Select(v["x_shape"], i) > 0,
                                Select(v["x_shape"], i) <= MAX_SZ_DIM
                            ))
                    for i in range(MAX_N_DIM)
                ]),

                # Same dtype as y
                v["x_dtype"] == v["y_dtype"],

                # Numeric dtype (exclude string types)
                And([
                    v["x_dtype"] != list_of_available_dtypes.index(dt)
                    for dt in list_of_string_values_tf
                ]),

                # Broadcasting compatibility with valid y
                And([
                    Or(
                        # dimension absent in both
                        And(i >= v["x_ndim"], i >= v["y_ndim"]),

                        # equal dimensions
                        And(i < v["x_ndim"], i < v["y_ndim"],
                            Select(v["x_shape"], i) == Select(v["y_shape"], i)),

                        # one of them is 1
                        And(i < v["x_ndim"], i < v["y_ndim"],
                            Or(
                                Select(v["x_shape"], i) == 1,
                                Select(v["y_shape"], i) == 1
                            ))
                    )
                    for i in range(MAX_N_DIM)
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

            v["x_dtype"] == v["y_dtype"],

            And([
                v["x_dtype"] != list_of_available_dtypes.index(dt)
                for dt in list_of_string_values_tf
            ]),

            And([
                Or(
                    And(i >= v["x_ndim"], i >= v["y_ndim"]),
                    And(i < v["x_ndim"], i < v["y_ndim"],
                        Select(v["x_shape"], i) == Select(v["y_shape"], i)),
                    And(i < v["x_ndim"], i < v["y_ndim"],
                        Or(
                            Select(v["x_shape"], i) == 1,
                            Select(v["y_shape"], i) == 1
                        ))
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
    x = next(iter(arg1.values()))
    y = next(iter(arg2.values()))

    # -------------------------
    # Invariant learning phase
    # -------------------------
    if not solver:
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            return False

        solver = Solver()

        # Declarations
        x_ndim = Int('x_ndim')
        y_ndim = Int('y_ndim')

        x_shape = Array('x_shape', IntSort(), IntSort())
        y_shape = Array('y_shape', IntSort(), IntSort())

        x_dtype = Int('x_dtype')
        y_dtype = Int('y_dtype')

        # Assign values
        solver.add(x_ndim == x.ndim)
        solver.add(y_ndim == y.ndim)

        for i in range(x.ndim):
            x_shape = Store(x_shape, i, x.shape[i])
        for i in range(y.ndim):
            y_shape = Store(y_shape, i, y.shape[i])

        solver.add(x_dtype == list_of_available_dtypes.index(x.dtype))
        solver.add(y_dtype == list_of_available_dtypes.index(y.dtype))

        # Apply rule
        rule_2(solver, {
            "x_ndim": x_ndim,
            "y_ndim": y_ndim,
            "x_shape": x_shape,
            "y_shape": y_shape,
            "x_dtype": x_dtype,
            "y_dtype": y_dtype
        })

        return solver.check() == sat

    # -------------------------
    # Fuzz generation phase
    # -------------------------
    else:
        rule_2(
            solver,
            {
                "x_ndim": x["ndim"],
                "y_ndim": y["ndim"],
                "x_shape": x["shape"],
                "y_shape": y["shape"],
                "x_dtype": x["dtype"],
                "y_dtype": y["dtype"]
            },
            neg
        )