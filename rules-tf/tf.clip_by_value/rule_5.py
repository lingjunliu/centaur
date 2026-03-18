import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 5: tf.clip_by_value tensor compatibility rule
# Assumes clip_value_min and clip_value_max are valid
"""
1. t has valid rank
2. All dimensions > 0
3. Same dtype across all three tensors
4. clip_value_min broadcastable to t
5. clip_value_max broadcastable to t
"""

rule_5 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # Valid rank
                v["t_ndim"] >= 0,
                v["min_ndim"] >= 0,
                v["max_ndim"] >= 0,

                # All dimensions of t must be positive
                And([
                    Implies(i < v["t_ndim"],
                            Select(v["t_shape"], i) > 0)
                    for i in range(MAX_N_DIM)
                ]),

                # Same dtype
                v["t_dtype"] == v["min_dtype"],
                v["t_dtype"] == v["max_dtype"],

                # Broadcasting: min -> t
                And([
                    Or(
                        And(i >= v["t_ndim"], i >= v["min_ndim"]),
                        And(i < v["t_ndim"], i < v["min_ndim"],
                            Select(v["t_shape"], i) == Select(v["min_shape"], i)),
                        And(i < v["t_ndim"], i < v["min_ndim"],
                            Select(v["min_shape"], i) == 1)
                    )
                    for i in range(MAX_N_DIM)
                ]),

                # Broadcasting: max -> t
                And([
                    Or(
                        And(i >= v["t_ndim"], i >= v["max_ndim"]),
                        And(i < v["t_ndim"], i < v["max_ndim"],
                            Select(v["t_shape"], i) == Select(v["max_shape"], i)),
                        And(i < v["t_ndim"], i < v["max_ndim"],
                            Select(v["max_shape"], i) == 1)
                    )
                    for i in range(MAX_N_DIM)
                ])
            )
        ) if n else
        And(
            v["t_ndim"] >= 0,
            v["min_ndim"] >= 0,
            v["max_ndim"] >= 0,

            And([
                Implies(i < v["t_ndim"],
                        Select(v["t_shape"], i) > 0)
                for i in range(MAX_N_DIM)
            ]),

            v["t_dtype"] == v["min_dtype"],
            v["t_dtype"] == v["max_dtype"],

            And([
                Or(
                    And(i >= v["t_ndim"], i >= v["min_ndim"]),
                    And(i < v["t_ndim"], i < v["min_ndim"],
                        Select(v["t_shape"], i) == Select(v["min_shape"], i)),
                    And(i < v["t_ndim"], i < v["min_ndim"],
                        Select(v["min_shape"], i) == 1)
                )
                for i in range(MAX_N_DIM)
            ]),

            And([
                Or(
                    And(i >= v["t_ndim"], i >= v["max_ndim"]),
                    And(i < v["t_ndim"], i < v["max_ndim"],
                        Select(v["t_shape"], i) == Select(v["max_shape"], i)),
                    And(i < v["t_ndim"], i < v["max_ndim"],
                        Select(v["max_shape"], i) == 1)
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_5_func(arg1, arg2, arg3, solver=None, neg=False):
    t = next(iter(arg1.values()))
    clip_min = next(iter(arg2.values()))
    clip_max = next(iter(arg3.values()))

    # -------------------------
    # Invariant learning phase
    # -------------------------
    if not solver:
        if not (isinstance(t, np.ndarray) and
                isinstance(clip_min, np.ndarray) and
                isinstance(clip_max, np.ndarray)):
            return False

        solver = Solver()

        # Variable declarations
        t_ndim = Int('t_ndim')
        min_ndim = Int('min_ndim')
        max_ndim = Int('max_ndim')

        t_shape = Array('t_shape', IntSort(), IntSort())
        min_shape = Array('min_shape', IntSort(), IntSort())
        max_shape = Array('max_shape', IntSort(), IntSort())

        t_dtype = Int('t_dtype')
        min_dtype = Int('min_dtype')
        max_dtype = Int('max_dtype')

        # Assign values
        solver.add(t_ndim == t.ndim)
        solver.add(min_ndim == clip_min.ndim)
        solver.add(max_ndim == clip_max.ndim)

        for i in range(t.ndim):
            t_shape = Store(t_shape, i, t.shape[i])
        for i in range(clip_min.ndim):
            min_shape = Store(min_shape, i, clip_min.shape[i])
        for i in range(clip_max.ndim):
            max_shape = Store(max_shape, i, clip_max.shape[i])

        solver.add(t_dtype == list_of_available_dtypes.index(t.dtype))
        solver.add(min_dtype == list_of_available_dtypes.index(clip_min.dtype))
        solver.add(max_dtype == list_of_available_dtypes.index(clip_max.dtype))

        # Apply rule
        rule_5(solver, {
            "t_ndim": t_ndim,
            "min_ndim": min_ndim,
            "max_ndim": max_ndim,
            "t_shape": t_shape,
            "min_shape": min_shape,
            "max_shape": max_shape,
            "t_dtype": t_dtype,
            "min_dtype": min_dtype,
            "max_dtype": max_dtype
        })

        return solver.check() == sat

    # -------------------------
    # Fuzz generation phase
    # -------------------------
    else:
        rule_5(
            solver,
            {
                "t_ndim": t["ndim"],
                "min_ndim": clip_min["ndim"],
                "max_ndim": clip_max["ndim"],
                "t_shape": t["shape"],
                "min_shape": clip_min["shape"],
                "max_shape": clip_max["shape"],
                "t_dtype": t["dtype"],
                "min_dtype": clip_min["dtype"],
                "max_dtype": clip_max["dtype"]
            },
            neg
        )