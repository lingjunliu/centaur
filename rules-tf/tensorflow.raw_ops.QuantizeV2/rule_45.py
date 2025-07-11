import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If axis is specified, min_range and max_range should be 1D tensors, axis should be a valid axis of input, and range tensors must be float tensors (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] != -1, (And(And(And(And(And(And(And(v["arg2_ndim"] == 1, v["arg3_ndim"] == 1), Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg4_value"])), Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], v["arg4_value"])), (-1 * v["arg1_ndim"]) <= v["arg4_value"]), v["arg4_value"] <= (v["arg1_ndim"] - 1)), v["arg2_dtype"] == 7), v["arg3_dtype"] == 7)), False)) if n else
          If(v["arg4_value"] != -1, (And(And(And(And(And(And(And(v["arg2_ndim"] == 1, v["arg3_ndim"] == 1), Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg4_value"])), Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], v["arg4_value"])), (-1 * v["arg1_ndim"]) <= v["arg4_value"]), v["arg4_value"] <= (v["arg1_ndim"] - 1)), v["arg2_dtype"] == 7), v["arg3_dtype"] == 7)), False))
)

def rule_45_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
