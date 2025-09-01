import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# To avoid `AttributeError: 'Tensor' object has no attribute '_forward_pre_hooks'`, we need v_1 be a Tensor, not a raw Tensor (for which the function doesn't work (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_shape"], 0) >= -1) if n else
          Select(v["arg1_shape"], 0) >= -1)
)

def rule_57_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 57
        rule_57(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_shape': arg1['shape']}, neg)
