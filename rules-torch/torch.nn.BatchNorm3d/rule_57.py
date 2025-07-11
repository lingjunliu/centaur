import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The number of features should be smaller than the total number of elements, but large enough. (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(And(1 < v["arg2_value"], v["arg2_value"] < Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) * Select(v["arg1_shape"], 4))) if n else
          And(1 < v["arg2_value"], v["arg2_value"] < Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) * Select(v["arg1_shape"], 4)))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 57
        rule_57(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
