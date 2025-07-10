import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Specific to only integer as a good base check condition, then there is one specific thing that does need to have be checked. (Rule 227)

rule_227 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, Select(v["arg2_shape"], 0) > 0, False)) if n else
          If(v["arg1_dtype"] == 1, Select(v["arg2_shape"], 0) > 0, False))
)

def rule_227_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 227
        rule_227(solver, {'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_227(solver, {'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape']}, neg)
