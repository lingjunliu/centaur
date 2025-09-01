import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Combination of all key parameter constraints (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(v["arg1_ndim"] >= 2, v["arg2_value"] > 0), v["arg2_value"] <= Select(v["arg1_shape"], 1)), Select(v["arg1_shape"], 1) % v["arg2_value"] == 0), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], 1)), v["arg4_value"] >= 0), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8)))) if n else
          And(And(And(And(And(And(And(v["arg1_ndim"] >= 2, v["arg2_value"] > 0), v["arg2_value"] <= Select(v["arg1_shape"], 1)), Select(v["arg1_shape"], 1) % v["arg2_value"] == 0), v["arg3_ndim"] == 1), Select(v["arg3_shape"], 0) == Select(v["arg1_shape"], 1)), v["arg4_value"] >= 0), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))))
)

def rule_28_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Real('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)

        # Constraints for rule 28
        rule_28(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value']}, neg)
