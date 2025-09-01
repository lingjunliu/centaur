import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the tensor has >1 dimension then num of groups must be positive, and must be a factor of channels and must not result index out of bounds (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 1, And(And(And(And(v["arg2_value"] > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 1) % v["arg2_value"] == 0), (1 < v["arg1_ndim"])), ((0 - v["arg1_ndim"]) <= 1)), True)) if n else
          If(v["arg1_ndim"] > 1, And(And(And(And(v["arg2_value"] > 0, Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 1) % v["arg2_value"] == 0), (1 < v["arg1_ndim"])), ((0 - v["arg1_ndim"]) <= 1)), True))
)

def rule_49_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 49
        rule_49(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
