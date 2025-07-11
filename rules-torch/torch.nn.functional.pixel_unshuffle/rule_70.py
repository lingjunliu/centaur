import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If the number of dimensions of the input tensor is less than 3, downscale_factor is 1. Other wise downscale_factor > 0, width and height are divisible by the downscale_factor (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] < 3, v["arg2_value"] == 1, And(And(v["arg2_value"] > 0, (Select(v["arg1_shape"], v["arg1_ndim"] - 2) % v["arg2_value"] == 0)), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) % v["arg2_value"] == 0)))) if n else
          If(v["arg1_ndim"] < 3, v["arg2_value"] == 1, And(And(v["arg2_value"] > 0, (Select(v["arg1_shape"], v["arg1_ndim"] - 2) % v["arg2_value"] == 0)), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) % v["arg2_value"] == 0))))
)

def rule_70_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 70
        rule_70(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
