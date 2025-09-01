import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# transformed dimensions of the tensor must be representable by int32 (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(And((Select(v["arg1_shape"], 1) * v["arg3_value"] - (Select(v["arg2_shape"], 0) + Select(v["arg2_shape"], 1))) < 2147483647, (Select(v["arg1_shape"], 2) * v["arg3_value"] - (Select(v["arg2_shape"], 2) + Select(v["arg2_shape"], 3))) < 2147483647)) if n else
          And((Select(v["arg1_shape"], 1) * v["arg3_value"] - (Select(v["arg2_shape"], 0) + Select(v["arg2_shape"], 1))) < 2147483647, (Select(v["arg1_shape"], 2) * v["arg3_value"] - (Select(v["arg2_shape"], 2) + Select(v["arg2_shape"], 3))) < 2147483647))
)

def rule_113_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 113
        rule_113(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value']}, neg)
