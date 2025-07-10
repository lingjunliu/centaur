import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Tensor's first dimension should be positive and smaller than 10 and v2 should be "relu" or "tanh" or "sigmoid" or "elu" (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 0) < 10), (Or(Or(Or(v["arg2_value"] == 12, v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 16)))) if n else
          And(And(Select(v["arg1_shape"], 0) > 0, Select(v["arg1_shape"], 0) < 10), (Or(Or(Or(v["arg2_value"] == 12, v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 16))))
)

def rule_74_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 74
        rule_74(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
