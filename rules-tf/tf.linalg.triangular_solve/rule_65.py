import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Matrices in the last dimensions must be square and have positive size (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 2) == Select(v["arg1_shape"], v["arg1_ndim"] - 1)), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0), Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0)) if n else
          And(And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], v["arg1_ndim"] - 2) == Select(v["arg1_shape"], v["arg1_ndim"] - 1)), Select(v["arg1_shape"], v["arg1_ndim"] - 1) > 0), Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0))
)

def rule_65_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 65
        rule_65(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
