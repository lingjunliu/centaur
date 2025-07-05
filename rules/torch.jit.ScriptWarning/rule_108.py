import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# warning if a tensor has a batch size of 1 and the rest are large, suggesting possible misuse of batching (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((v["arg1_ndim"] == 4), (Select(v["arg1_shape"], 0) == 1)), (Select(v["arg1_shape"], 1) > 100)), (Select(v["arg1_shape"], 2) > 100)), (Select(v["arg1_shape"], 3) > 100))) if n else
          And(And(And(And((v["arg1_ndim"] == 4), (Select(v["arg1_shape"], 0) == 1)), (Select(v["arg1_shape"], 1) > 100)), (Select(v["arg1_shape"], 2) > 100)), (Select(v["arg1_shape"], 3) > 100)))
)

def rule_108_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 108
        rule_108(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
