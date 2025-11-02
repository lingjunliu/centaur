import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x is a boolean tensor and y is a scalar, z will have the same shape as x (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_dtype"] == 0, v["arg2_ndim"] == 0), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0), True)) if n else
          If(And(v["arg1_dtype"] == 0, v["arg2_ndim"] == 0), Select(v["arg1_shape"], 0) == Select(v["arg3_shape"], 0), True))
)

def rule_65_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 65
        rule_65(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape']}, neg)
