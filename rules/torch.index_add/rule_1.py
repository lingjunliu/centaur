import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# input and source tensors should have the same dtype, index tensor should have int32 or int64 dtype, and index values should be within bounds (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_dtype"] == v["arg4_dtype"], (Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4))), Select(v["arg3_range"], 0) >= 0), Select(v["arg3_range"], 1) < Select(v["arg1_shape"], v["arg2_value"]))) if n else
          And(And(And(v["arg1_dtype"] == v["arg4_dtype"], (Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4))), Select(v["arg3_range"], 0) >= 0), Select(v["arg3_range"], 1) < Select(v["arg1_shape"], v["arg2_value"])))
)

def rule_1_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 1
        rule_1(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_range': arg3_range, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_range': arg3['range'], 'arg4_dtype': arg4['dtype']}, neg)
