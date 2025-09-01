import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# fft_length shape must be compatible with subtraction operation (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_ndim"] == 3, v["arg2_ndim"] == 1), Select(v["arg2_shape"], 0) == 3)) if n else
          And(And(v["arg1_ndim"] == 3, v["arg2_ndim"] == 1), Select(v["arg2_shape"], 0) == 3))
)

def rule_24_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 24
        rule_24(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
