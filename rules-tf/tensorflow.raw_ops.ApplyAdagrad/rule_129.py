import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if shape of var is greater than grad, then accum must be type float (Rule 129)

rule_129 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > Select(v["arg2_shape"], 0), Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), False)) if n else
          If(Select(v["arg1_shape"], 0) > Select(v["arg2_shape"], 0), Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8), False))
)

def rule_129_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 129
        rule_129(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_129(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_dtype': arg3['dtype']}, neg)
