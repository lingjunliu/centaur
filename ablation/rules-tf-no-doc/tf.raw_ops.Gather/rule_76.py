import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The shape of the index tensor along dimension 0 must be less or equal than 2^31-1. If indices is not int32 or int64 then skip (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(Or((v["arg2_dtype"] == 3), (v["arg2_dtype"] == 4)), Select(v["arg1_shape"], 0) <= 2147483647, True)) if n else
          If(Or((v["arg2_dtype"] == 3), (v["arg2_dtype"] == 4)), Select(v["arg1_shape"], 0) <= 2147483647, True))
)

def rule_76_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 76
        rule_76(solver, {'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype']}, neg)
