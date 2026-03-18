import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule: constraints for tf.math.square input tensor

rule_2 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                v["x_ndim"] >= 0,
                v["x_ndim"] <= MAX_N_DIM,
                And([
                    Implies(i < v["x_ndim"],
                            And(
                                Select(v["x_shape"], i) >= 0,
                                Select(v["x_shape"], i) <= MAX_SZ_DIM
                            ))
                    for i in range(MAX_N_DIM)
                ]),
                v["x_numel"] >= 0,
                v["x_numel"] <= MAX_SZ_NUM,
                v["x_dtype_valid"]
            )
        )
    ) if n else
    s.add(
        And(
            v["x_ndim"] >= 0,
            v["x_ndim"] <= MAX_N_DIM,
            And([
                Implies(i < v["x_ndim"],
                        And(
                            Select(v["x_shape"], i) >= 0,
                            Select(v["x_shape"], i) <= MAX_SZ_DIM
                        ))
                for i in range(MAX_N_DIM)
            ]),
            v["x_numel"] >= 0,
            v["x_numel"] <= MAX_SZ_NUM,
            v["x_dtype_valid"]
        )
    )
)
def rule_2_func(arg1, solver=None, neg=False):
    x = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(x, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        x_ndim = Int('x_ndim')
        x_shape = Array('x_shape', IntSort(), IntSort())
        x_numel = Int('x_numel')
        x_dtype_valid = Bool('x_dtype_valid')

        # Assign concrete values
        solver.add(x_ndim == x.ndim)
        solver.add(x_numel == x.size)

        for i in range(x.ndim):
            solver.add(Select(x_shape, i) == x.shape[i])

        # Valid numeric dtypes
        numeric_types = [
            np.int8, np.int16, np.int32, np.int64,
            np.float16, np.float32, np.float64,
            np.complex64, np.complex128
        ]

        solver.add(x_dtype_valid == (x.dtype.type in numeric_types))

        # Apply rule
        rule_2(solver, {
            "x_ndim": x_ndim,
            "x_shape": x_shape,
            "x_numel": x_numel,
            "x_dtype_valid": x_dtype_valid
        })

        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(
            solver,
            {
                "x_ndim": x["ndim"],
                "x_shape": x["shape"],
                "x_numel": x["numel"],
                "x_dtype_valid": x["dtype_valid"]
            },
            neg
        )