import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# stride must be positive and reasonable value, must smaller than the input size. Must smaller or equal to kernel size and not zero and smaller than the input length, not a very small value, and not equal to the input length and should be greater than zero, and not larger than the input size. Stride cannot be zero either (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(v["arg1_value"] > 0, v["arg1_value"] < 1000), Select(v["arg2_shape"], v["arg2_ndim"] - 1) > v["arg1_value"]), v["arg1_value"] <= v["arg3_value"]), v["arg1_value"] < Select(v["arg2_shape"], v["arg2_ndim"] - 1)), v["arg1_value"] > 0), v["arg1_value"] != Select(v["arg2_shape"], v["arg2_ndim"] - 1)), v["arg1_value"] > 0), v["arg1_value"] < Select(v["arg2_shape"], v["arg2_ndim"] - 1)), v["arg1_value"] != 0)) if n else
          And(And(And(And(And(And(And(And(And(v["arg1_value"] > 0, v["arg1_value"] < 1000), Select(v["arg2_shape"], v["arg2_ndim"] - 1) > v["arg1_value"]), v["arg1_value"] <= v["arg3_value"]), v["arg1_value"] < Select(v["arg2_shape"], v["arg2_ndim"] - 1)), v["arg1_value"] > 0), v["arg1_value"] != Select(v["arg2_shape"], v["arg2_ndim"] - 1)), v["arg1_value"] > 0), v["arg1_value"] < Select(v["arg2_shape"], v["arg2_ndim"] - 1)), v["arg1_value"] != 0))
)

def rule_111_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 111
        rule_111(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
