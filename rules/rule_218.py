import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if tensor data type is complex128 and tensor shape[0]>0 , then the max should be greater than or equal to 1. (Rule 218)

rule_218 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 10, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_range"], 1) >= 1, False)) if n else
          If(And(v["arg1_dtype"] == 10, Select(v["arg1_shape"], 0) > 0), Select(v["arg1_range"], 1) >= 1, False))
)

def rule_218_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 218
        rule_218(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_218(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']}, neg)
