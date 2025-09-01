import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Quant max is less than 200 and axis can't be negative and scale can't have negative values and input must be 3D AND zero_point is non-empty AND quant_max must be greater than 50 AND scale and zero_point must have the same number of elements AND the sum of quant min and quant max > 100 AND zero_point can't have negative values (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(v["arg6_value"] < 200, v["arg4_value"] >= 0), Select(v["arg2_range"], 0) >= 0), v["arg1_ndim"] == 3), Select(v["arg3_shape"], 0) > 0), v["arg6_value"] > 50), Select(v["arg2_shape"], 0) == Select(v["arg3_shape"], 0)), v["arg5_value"] + v["arg6_value"] > 100), Select(v["arg3_range"], 0) >= 0)) if n else
          And(And(And(And(And(And(And(And(v["arg6_value"] < 200, v["arg4_value"] >= 0), Select(v["arg2_range"], 0) >= 0), v["arg1_ndim"] == 3), Select(v["arg3_shape"], 0) > 0), v["arg6_value"] > 50), Select(v["arg2_shape"], 0) == Select(v["arg3_shape"], 0)), v["arg5_value"] + v["arg6_value"] > 100), Select(v["arg3_range"], 0) >= 0))
)

def rule_97_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

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
        if not (isinstance(arg6, (int, np.integer)) and not isinstance(arg6, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 97
        rule_97(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg3_shape': arg3_shape, 'arg3_range': arg3_range, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg3_shape': arg3['shape'], 'arg3_range': arg3['range'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
