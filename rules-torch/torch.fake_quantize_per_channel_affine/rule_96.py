import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Zero point must be Int32, Float or Half and Quant min must be positive and Quant max greater than 10 and axis less than input ndim AND scale has two dims AND zero_point has less than 5 elements AND quant_min less than 50 AND scale can't have more than 5 elements AND input must have less than 4 dimensions (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And((Or(Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 6), v["arg3_dtype"] == 7)), v["arg5_value"] > 0), v["arg6_value"] > 10), v["arg4_value"] < v["arg1_ndim"]), v["arg2_ndim"] == 2), Select(v["arg3_shape"], 0) < 5), v["arg5_value"] < 50), Select(v["arg2_shape"], 0) < 5), v["arg1_ndim"] < 4)) if n else
          And(And(And(And(And(And(And(And((Or(Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 6), v["arg3_dtype"] == 7)), v["arg5_value"] > 0), v["arg6_value"] > 10), v["arg4_value"] < v["arg1_ndim"]), v["arg2_ndim"] == 2), Select(v["arg3_shape"], 0) < 5), v["arg5_value"] < 50), Select(v["arg2_shape"], 0) < 5), v["arg1_ndim"] < 4))
)

def rule_96_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 96
        rule_96(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
