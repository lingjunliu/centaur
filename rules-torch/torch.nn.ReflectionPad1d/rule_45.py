import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Check for integer overflow on the number of elements in the tensor - individual dimensions are reasonably small (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 2, And(Select(v["arg1_shape"], 0) < 1000, Select(v["arg1_shape"], 1) < 1000), If(v["arg1_ndim"] == 3, And(And(Select(v["arg1_shape"], 0) < 100, Select(v["arg1_shape"], 1) < 100), Select(v["arg1_shape"], 2) < 100), False))) if n else
          If(v["arg1_ndim"] == 2, And(Select(v["arg1_shape"], 0) < 1000, Select(v["arg1_shape"], 1) < 1000), If(v["arg1_ndim"] == 3, And(And(Select(v["arg1_shape"], 0) < 100, Select(v["arg1_shape"], 1) < 100), Select(v["arg1_shape"], 2) < 100), False)))
)

def rule_45_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
