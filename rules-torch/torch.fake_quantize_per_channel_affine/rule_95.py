import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Scale must be float and 1D and input tensor ndim > 2 and Zero point must have more than one element AND quant_min greater than -10 AND input and scale must have the same dtype AND axis must be non-negative and zero_point's dtype can't be bool AND quant_max is greater than zero (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(v["arg2_dtype"] == 7, v["arg2_ndim"] == 1), v["arg1_ndim"] > 2), Select(v["arg3_shape"], 0) > 1), v["arg5_value"] > -10), v["arg1_dtype"] == v["arg2_dtype"]), v["arg4_value"] >= 0), v["arg3_dtype"] != 0), v["arg6_value"] > 0)) if n else
          And(And(And(And(And(And(And(And(v["arg2_dtype"] == 7, v["arg2_ndim"] == 1), v["arg1_ndim"] > 2), Select(v["arg3_shape"], 0) > 1), v["arg5_value"] > -10), v["arg1_dtype"] == v["arg2_dtype"]), v["arg4_value"] >= 0), v["arg3_dtype"] != 0), v["arg6_value"] > 0))
)

def rule_95_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Int('arg4_value')
        arg5_value = Int('arg5_value')
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == int(arg4))
        solver.add(arg5_value == int(arg5))
        solver.add(arg6_value == int(arg6))

        # Constraints for rule 95
        rule_95(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
