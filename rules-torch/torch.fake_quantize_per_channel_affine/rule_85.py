import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Quant max is less than 200 and axis can't be negative and scale can't have negative values and input must be 3D AND zero_point is non-empty (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg5_value"] < 200, v["arg4_value"] >= 0), Select(v["arg2_range"], 0) >= 0), v["arg1_ndim"] == 3), Select(v["arg3_shape"], 0) > 0)) if n else
          And(And(And(And(v["arg5_value"] < 200, v["arg4_value"] >= 0), Select(v["arg2_range"], 0) >= 0), v["arg1_ndim"] == 3), Select(v["arg3_shape"], 0) > 0))
)

def rule_85_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

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
        if not (isinstance(arg5, (int, np.integer)) and not isinstance(arg5, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))

        # Constraints for rule 85
        rule_85(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
