import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# LU input tensor must have float or complex type, and shape[0] of the pivots must be greater than 0 (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), Select(v["arg2_shape"], 0) > 0)) if n else
          And((Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), Select(v["arg2_shape"], 0) > 0))
)

def rule_75_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 75
        rule_75(solver, {'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape']}, neg)
