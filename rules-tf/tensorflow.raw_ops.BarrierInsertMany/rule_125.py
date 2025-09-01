import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If Keys tensor is integer then max value of keys tensor must be lesser than 10000 and component index must be even and must not be zero and max of handle must be more than 5 and min of values must not be more than 1000 and values must have greater number of axis than keys and values and keys' dtypes should not match and its name should not be same (Rule 125)

rule_125 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_dtype"] < 11), (v["arg2_value"] % 2 == 0)), (v["arg2_value"] != 0)), And(And(And(And(And(Select(v["arg1_range"], 1) < 10000, Select(v["arg3_range"], 1) > 5), Select(v["arg4_range"], 0) < 1000), v["arg4_ndim"] > v["arg1_ndim"]), v["arg4_dtype"] != v["arg1_dtype"]), v["arg5_value"] != v["arg5_value"]), True)) if n else
          If(And(And((v["arg1_dtype"] < 11), (v["arg2_value"] % 2 == 0)), (v["arg2_value"] != 0)), And(And(And(And(And(Select(v["arg1_range"], 1) < 10000, Select(v["arg3_range"], 1) > 5), Select(v["arg4_range"], 0) < 1000), v["arg4_ndim"] > v["arg1_ndim"]), v["arg4_dtype"] != v["arg1_dtype"]), v["arg5_value"] != v["arg5_value"]), True))
)

def rule_125_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_ndim = Int('arg4_ndim')
        arg4_dtype = Int('arg4_dtype')
        arg4_range = Array('arg4_range', IntSort(), IntSort())
        arg5_value = String('arg5_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        arg4_range = Store(arg4_range, 0, int(np.min(arg4)))
        arg4_range = Store(arg4_range, 1, int(np.max(arg4)))
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 125
        rule_125(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_range': arg3_range, 'arg4_range': arg4_range, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_125(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_range': arg3['range'], 'arg4_range': arg4['range'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype'], 'arg5_value': arg5['value']}, neg)
