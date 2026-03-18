import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *
# Rule 6: Structural constraints for tf.gather
"""
1. 0 ≤ axis < rank(params)
2. 0 ≤ batch_dims ≤ axis
3. rank(indices) ≥ batch_dims
4. For all i < batch_dims: params.shape[i] == indices.shape[i]
5. If validate_indices=True, then: 0 ≤ indices < params.shape[axis]
6. Output rank: rank(params) - 1 - axis + rank(indices)
"""
rule_6 = lambda s, v, n=False: (
    s.add(
        Not(
            And(
                # axis in valid range
                v["axis"] >= 0,
                v["axis"] < v["params_ndim"],

                # batch_dims valid
                v["batch_dims"] >= 0,
                v["batch_dims"] <= v["axis"],

                # indices rank >= batch_dims
                v["indices_ndim"] >= v["batch_dims"],

                # batch dimensions must match
                And([
                    Implies(
                        i < v["batch_dims"],
                        Select(v["params_shape"], i) ==
                        Select(v["indices_shape"], i)
                    )
                    for i in range(MAX_N_DIM)
                ])
            )
        ) if n else
        And(
            v["axis"] >= 0,
            v["axis"] < v["params_ndim"],

            v["batch_dims"] >= 0,
            v["batch_dims"] <= v["axis"],

            v["indices_ndim"] >= v["batch_dims"],

            And([
                Implies(
                    i < v["batch_dims"],
                    Select(v["params_shape"], i) ==
                    Select(v["indices_shape"], i)
                )
                for i in range(MAX_N_DIM)
            ])
        )
    )
)

def rule_6_func(arg1, arg2, arg3, arg4, solver=None, neg=False):

    params = next(iter(arg1.values()))
    indices = next(iter(arg2.values()))
    axis = next(iter(arg3.values()))
    batch_dims = next(iter(arg4.values()))

    # -----------------------------
    # Invariant learning phase
    # -----------------------------
    if not solver:

        if not isinstance(params, np.ndarray):
            return False
        if not isinstance(indices, np.ndarray):
            return False

        solver = Solver()

        # Declare symbolic variables
        params_ndim = Int('params_ndim')
        indices_ndim = Int('indices_ndim')

        params_shape = Array('params_shape', IntSort(), IntSort())
        indices_shape = Array('indices_shape', IntSort(), IntSort())

        axis_sym = Int('axis')
        batch_dims_sym = Int('batch_dims')

        # Assign concrete values
        solver.add(params_ndim == params.ndim)
        solver.add(indices_ndim == indices.ndim)
        solver.add(axis_sym == axis)
        solver.add(batch_dims_sym == batch_dims)

        for i in range(params.ndim):
            params_shape = Store(params_shape, i, params.shape[i])

        for i in range(indices.ndim):
            indices_shape = Store(indices_shape, i, indices.shape[i])

        # Apply rule
        rule_6(
            solver,
            {
                "params_ndim": params_ndim,
                "indices_ndim": indices_ndim,
                "params_shape": params_shape,
                "indices_shape": indices_shape,
                "axis": axis_sym,
                "batch_dims": batch_dims_sym
            }
        )

        return solver.check() == sat

    # -----------------------------
    # Fuzz generation phase
    # -----------------------------
    else:
        rule_6(
            solver,
            {
                "params_ndim": params["ndim"],
                "indices_ndim": indices["ndim"],
                "params_shape": params["shape"],
                "indices_shape": indices["shape"],
                "axis": axis,
                "batch_dims": batch_dims
            },
            neg
        )