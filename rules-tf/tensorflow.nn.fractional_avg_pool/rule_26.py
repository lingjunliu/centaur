import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If overlapping is true then the output size will be smaller. (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(Select(v["arg2_shape"], 1) / Select(v["arg3_values"], 1) < Select(v["arg2_shape"], 1), Select(v["arg2_shape"], 2) / Select(v["arg3_values"], 2) < Select(v["arg2_shape"], 2)), False)) if n else
          If(v["arg1_value"] == True, And(Select(v["arg2_shape"], 1) / Select(v["arg3_values"], 1) < Select(v["arg2_shape"], 1), Select(v["arg2_shape"], 2) / Select(v["arg3_values"], 2) < Select(v["arg2_shape"], 2)), False))
)

def rule_26_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all(isinstance(e, (float, np.floating)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 26
        rule_26(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values']}, neg)
