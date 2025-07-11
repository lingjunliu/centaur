import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the rank of data is greater than 1, then segment_ids length must equal indices length (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 1, Select(v["arg3_shape"], 0) == Select(v["arg2_shape"], 0), False)) if n else
          If(v["arg1_ndim"] > 1, Select(v["arg3_shape"], 0) == Select(v["arg2_shape"], 0), False))
)

def rule_63_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 63
        rule_63(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape']}, neg)
