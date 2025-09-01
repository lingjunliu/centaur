import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The shape of seq_length must be [batch_size] or [] where [batch_size] is the first dimension of labels to prevent incompatible shape errors during processing. (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg2_ndim"] == 1, Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), (v["arg2_ndim"] == 0))) if n else
          Or((And(v["arg2_ndim"] == 1, Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))), (v["arg2_ndim"] == 0)))
)

def rule_28_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 28
        rule_28(solver, {'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
