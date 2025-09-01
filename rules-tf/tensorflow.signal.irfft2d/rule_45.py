import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input tensor must be complex64 or complex128 and fft_length must be int32 and its shape must be [2] if it is provided (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11)), (If(v["arg2_ndim"] > 0, And(And(v["arg2_dtype"] == 3, v["arg2_ndim"] == 1), Select(v["arg2_shape"], 0) == 2), True)))) if n else
          And((Or(v["arg1_dtype"] == 10, v["arg1_dtype"] == 11)), (If(v["arg2_ndim"] > 0, And(And(v["arg2_dtype"] == 3, v["arg2_ndim"] == 1), Select(v["arg2_shape"], 0) == 2), True))))
)

def rule_45_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 45
        rule_45(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape']}, neg)
