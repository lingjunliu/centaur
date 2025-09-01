import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the keys is string, then the max value of component index should be less than 1024 and it should be greater than -1024 and values should have length greater than component_index and ndim of handle should be 0 or 1 and component index must be more than 1 and dtype of values should not be default (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, And(And(And(And(And(v["arg2_value"] < 1024, v["arg2_value"] > -1024), Select(v["arg3_shape"], 0) > v["arg2_value"]), (Or(v["arg4_ndim"] == 0, v["arg4_ndim"] == 1))), v["arg2_value"] > 1), v["arg3_dtype"] != 12), True)) if n else
          If(v["arg1_dtype"] == 11, And(And(And(And(And(v["arg2_value"] < 1024, v["arg2_value"] > -1024), Select(v["arg3_shape"], 0) > v["arg2_value"]), (Or(v["arg4_ndim"] == 0, v["arg4_ndim"] == 1))), v["arg2_value"] > 1), v["arg3_dtype"] != 12), True))
)

def rule_113_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 113
        rule_113(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape, 'arg3_dtype': arg3_dtype, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape'], 'arg3_dtype': arg3['dtype'], 'arg4_ndim': arg4['ndim']}, neg)
