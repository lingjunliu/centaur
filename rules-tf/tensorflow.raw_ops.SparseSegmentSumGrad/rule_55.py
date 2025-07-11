import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape of indices is greater than 0 then segment_ids should be less than the maximum value for its dtype (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) > 0, If(v["arg1_dtype"] == 3, Select(v["arg1_range"], 1) < 2147483647, Select(v["arg1_range"], 1) < 9223372036854775807), False)) if n else
          If(Select(v["arg2_shape"], 0) > 0, If(v["arg1_dtype"] == 3, Select(v["arg1_range"], 1) < 2147483647, Select(v["arg1_range"], 1) < 9223372036854775807), False))
)

def rule_55_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 55
        rule_55(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg2_shape': arg2['shape']}, neg)
