import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Power (p (Rule 136)

rule_136 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > 0, v["arg1_value"] < 100), Select(v["arg2_shape"], 0) > 0)) if n else
          And(And(v["arg1_value"] > 0, v["arg1_value"] < 100), Select(v["arg2_shape"], 0) > 0))
)

def rule_136_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 136
        rule_136(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_136(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape']}, neg)
