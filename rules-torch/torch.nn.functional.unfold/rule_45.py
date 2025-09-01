import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dilation must not be excessive relative to kernel_size and input size - stricter version and check for zero kernel (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0), And(And((Select(v["arg2_values"], 0) - 1) * v["arg1_value"] + 1 < Select(v["arg3_shape"], 2), (Select(v["arg2_values"], 1) - 1) * v["arg1_value"] + 1 < Select(v["arg3_shape"], 3)), v["arg1_value"] < 100), True)) if n else
          If(And(Select(v["arg2_values"], 0) > 0, Select(v["arg2_values"], 1) > 0), And(And((Select(v["arg2_values"], 0) - 1) * v["arg1_value"] + 1 < Select(v["arg3_shape"], 2), (Select(v["arg2_values"], 1) - 1) * v["arg1_value"] + 1 < Select(v["arg3_shape"], 3)), v["arg1_value"] < 100), True))
)

def rule_45_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape']}, neg)
