import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If overlapping is enabled the resulting output must be smaller than the input (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, And(And(And(Select(v["arg1_shape"], 1) > 0, Select(v["arg1_shape"], 2) > 0), Select(v["arg3_values"], 1) > 1.0), Select(v["arg3_values"], 2) > 1.0), False)) if n else
          If(v["arg2_value"] == True, And(And(And(Select(v["arg1_shape"], 1) > 0, Select(v["arg1_shape"], 2) > 0), Select(v["arg3_values"], 1) > 1.0), Select(v["arg3_values"], 2) > 1.0), False))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, list) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Bool('arg2_value')
        arg3_values = Array('arg3_values', IntSort(), RealSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == arg2)
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 39
        rule_39(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values']}, neg)
