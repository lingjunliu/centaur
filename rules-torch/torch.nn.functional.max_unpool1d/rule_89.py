import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Data type relationship enforcement and constraints of params relative to input with respect to size compatibility (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg2_dtype"] == 4, (If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 6), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 6), If(And(7 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), And(7 <= v["arg2_dtype"], v["arg2_dtype"] <= 11), True)))), (If(v["arg3_length"] == 1, (Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg6_value"] - v["arg4_value"]) / v["arg5_value"] + 1 == Select(v["arg3_values"], 0), True))), Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg6_value"] >= v["arg4_value"]), Select(v["arg1_shape"], v["arg1_ndim"] - 1) >= v["arg5_value"])) if n else
          And(And(And(And(v["arg2_dtype"] == 4, (If(And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 6), And(1 <= v["arg2_dtype"], v["arg2_dtype"] <= 6), If(And(7 <= v["arg1_dtype"], v["arg1_dtype"] <= 11), And(7 <= v["arg2_dtype"], v["arg2_dtype"] <= 11), True)))), (If(v["arg3_length"] == 1, (Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg6_value"] - v["arg4_value"]) / v["arg5_value"] + 1 == Select(v["arg3_values"], 0), True))), Select(v["arg1_shape"], v["arg1_ndim"] - 1) + 2 * v["arg6_value"] >= v["arg4_value"]), Select(v["arg1_shape"], v["arg1_ndim"] - 1) >= v["arg5_value"]))
)

def rule_89_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
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
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 89
        rule_89(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg3_length': arg3_length, 'arg3_values': arg3_values, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
