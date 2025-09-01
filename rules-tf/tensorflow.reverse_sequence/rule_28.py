import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If seq_lengths is specified, it's maximum value must be less or equal to input's seq_axis dimension and it is a valid int32/int64 tensor (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, And(Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], v["arg3_value"]), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), True)) if n else
          If(v["arg2_ndim"] > 0, And(Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], v["arg3_value"]), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), True))
)

def rule_28_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 28
        rule_28(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
