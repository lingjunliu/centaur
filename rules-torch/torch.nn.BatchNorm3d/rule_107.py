import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# number of features should be limited relative to the size of the tensor to prevent "concrete integer" error (Rule 107)

rule_107 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 100, Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3) * Select(v["arg2_shape"], 4) > 50, False)) if n else
          If(v["arg1_value"] > 100, Select(v["arg2_shape"], 2) * Select(v["arg2_shape"], 3) * Select(v["arg2_shape"], 4) > 50, False))
)

def rule_107_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 107
        rule_107(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)
