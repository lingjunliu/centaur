import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# For performance reasons, avoid very large dimensions on the innermost dimensions (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_shape"], v["arg1_ndim"] - 1) * Select(v["arg1_shape"], v["arg1_ndim"] - 2) * Select(v["arg1_shape"], v["arg1_ndim"] - 3) < 2000000) if n else
          Select(v["arg1_shape"], v["arg1_ndim"] - 1) * Select(v["arg1_shape"], v["arg1_ndim"] - 2) * Select(v["arg1_shape"], v["arg1_ndim"] - 3) < 2000000)
)

def rule_29_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 29
        rule_29(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
