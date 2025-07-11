import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If overlapping is disabled, then the pooling ratio can't be too big or the output will be too small. (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, And(Select(v["arg3_shape"], 1) / Select(v["arg2_values"], 1) > 1, Select(v["arg3_shape"], 2) / Select(v["arg2_values"], 2) > 1), False)) if n else
          If(v["arg1_value"] == False, And(Select(v["arg3_shape"], 1) / Select(v["arg2_values"], 1) > 1, Select(v["arg3_shape"], 2) / Select(v["arg2_values"], 2) > 1), False))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), RealSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 48
        rule_48(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape']}, neg)
