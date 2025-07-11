import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# The tensor must have either 2, 3, or 4 dimensions, the sum of the shapes should be greater than 0, and product of shapes should also be greater than 0. (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(Or(v["arg1_ndim"] == 2, v["arg1_ndim"] == 3), v["arg1_ndim"] == 4)), (Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], If(v["arg1_ndim"] > 1, 1, 0)) + Select(v["arg1_shape"], If(v["arg1_ndim"] > 2, 2, 0)) > 0)), (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], If(v["arg1_ndim"] > 1, 1, 0)) * Select(v["arg1_shape"], If(v["arg1_ndim"] > 2, 2, 0)) > 0))) if n else
          And(And((Or(Or(v["arg1_ndim"] == 2, v["arg1_ndim"] == 3), v["arg1_ndim"] == 4)), (Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], If(v["arg1_ndim"] > 1, 1, 0)) + Select(v["arg1_shape"], If(v["arg1_ndim"] > 2, 2, 0)) > 0)), (Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], If(v["arg1_ndim"] > 1, 1, 0)) * Select(v["arg1_shape"], If(v["arg1_ndim"] > 2, 2, 0)) > 0)))
)

def rule_36_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 36
        rule_36(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
