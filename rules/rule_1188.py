import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If scale factor is a tensor of values and one of the dimension is not specified, its data type must be an integer type (0 <= dtype <=5 (Rule 1188)

rule_1188 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_ndim"] > 0, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]))), (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), False)) if n else
          If(And(v["arg2_ndim"] > 0, (Or([And(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]))), (And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5)), False))
)

def rule_1188_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 1188
        rule_1188(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1188(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
