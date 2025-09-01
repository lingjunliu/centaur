import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel_size and dilation must result in a valid receptive field (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And((Select(v["arg1_values"], 0) - 1) * v["arg2_value"] < Select(v["arg3_shape"], 2), (Select(v["arg1_values"], 1) - 1) * v["arg2_value"] < Select(v["arg3_shape"], 3))) if n else
          And((Select(v["arg1_values"], 0) - 1) * v["arg2_value"] < Select(v["arg3_shape"], 2), (Select(v["arg1_values"], 1) - 1) * v["arg2_value"] < Select(v["arg3_shape"], 3)))
)

def rule_24_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 24
        rule_24(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_shape': arg3['shape']}, neg)
