import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
valid_fft3d_dtypes = ["complex64", "complex128"]

rule_88 = lambda s, v, n=False: (
    s.add(Not(
        And(
            # Rank constraint
            v["input_ndim"] >= 3,
            v["input_ndim"] <= MAX_N_DIM,

            # Last three dimensions must be positive
            Select(v["input_shape"], v["input_ndim"] - 1) > 0,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,
            Select(v["input_shape"], v["input_ndim"] - 3) > 0,

            # Complex dtype only
            Or([v["input_dtype"] == StringVal(dt) for dt in valid_fft3d_dtypes])
        )
    )) if n else
    s.add(
        And(
            v["input_ndim"] >= 3,
            v["input_ndim"] <= MAX_N_DIM,
            Select(v["input_shape"], v["input_ndim"] - 1) > 0,
            Select(v["input_shape"], v["input_ndim"] - 2) > 0,
            Select(v["input_shape"], v["input_ndim"] - 3) > 0,
            Or([v["input_dtype"] == StringVal(dt) for dt in valid_fft3d_dtypes])
        )
    )
)
def rule_88_func(arg1, solver=None, neg=False):

    input_tensor = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(input_tensor, np.ndarray):
            return False

        solver = Solver()

        input_ndim = Int('input_ndim')
        input_shape = Array('input_shape', IntSort(), IntSort())
        input_dtype = String('input_dtype')

        # Assign actual values
        solver.add(input_ndim == input_tensor.ndim)
        solver.add(input_dtype == StringVal(str(input_tensor.dtype)))

        for i in range(input_tensor.ndim):
            input_shape = Store(input_shape, i, input_tensor.shape[i])

        # Apply rule
        rule_88(solver, {
            "input_ndim": input_ndim,
            "input_shape": input_shape,
            "input_dtype": input_dtype
        })

        return solver.check() == sat

    # Fuzz generation phase
    else:
        rule_88(solver, {
            "input_ndim": input_tensor["ndim"],
            "input_shape": input_tensor["shape"],
            "input_dtype": input_tensor["dtype"]
        }, neg)