import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When padding_mode is "circular", padding in each dimension must be smaller than half of the respective input size (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 24, Or((And(And(v["arg2_length"] == 1, Select(v["arg2_values"], 0) < Select(v["arg3_shape"], 2) / 2), Select(v["arg2_values"], 0) < Select(v["arg3_shape"], 3) / 2)), (And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) < Select(v["arg3_shape"], 2) / 2), Select(v["arg2_values"], 1) < Select(v["arg3_shape"], 3) / 2))), True)) if n else
          If(v["arg1_value"] == 24, Or((And(And(v["arg2_length"] == 1, Select(v["arg2_values"], 0) < Select(v["arg3_shape"], 2) / 2), Select(v["arg2_values"], 0) < Select(v["arg3_shape"], 3) / 2)), (And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) < Select(v["arg3_shape"], 2) / 2), Select(v["arg2_values"], 1) < Select(v["arg3_shape"], 3) / 2))), True))
)

def rule_36_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_shape': arg3['shape']}, neg)
