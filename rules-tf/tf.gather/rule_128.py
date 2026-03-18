import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 128: params must support valid indices at axis=0
"""
rank(params) ≥ 1
params.shape[0] > max(indices)
params.shape[0] > 0
"""
rule_128 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # params must have at least 1 dimension
                v["params_ndim"] >= 1,

                # first dimension positive
                Select(v["params_shape"], 0) > 0,

                # must accommodate max index
                Select(v["params_shape"], 0) > v["indices_max"]
            )
        ) if n else
        And(
            v["params_ndim"] >= 1,
            Select(v["params_shape"], 0) > 0,
            Select(v["params_shape"], 0) > v["indices_max"]
        )
    )
)
def rule_128_func(arg1, arg2, solver=None, neg=False):

    params = next(iter(arg1.values()))
    indices = next(iter(arg2.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(params, np.ndarray):
            return False
        if not isinstance(indices, np.ndarray):
            return False

        solver = Solver()

        # Symbolic variables
        params_ndim = Int('params_ndim')
        params_shape = Array('params_shape', IntSort(), IntSort())
        indices_max = Int('indices_max')

        # Assign values
        solver.add(params_ndim == params.ndim)

        for i in range(params.ndim):
            params_shape = Store(params_shape, i, params.shape[i])

        solver.add(indices_max == int(np.max(indices)))

        # Apply rule
        rule_128(
            solver,
            {
                "params_ndim": params_ndim,
                "params_shape": params_shape,
                "indices_max": indices_max
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz input generation phase
    # -----------------------------
    else:
        rule_128(
            solver,
            {
                "params_ndim": params["ndim"],
                "params_shape": params["shape"],
                "indices_max": indices["max"]
            },
            neg
        )