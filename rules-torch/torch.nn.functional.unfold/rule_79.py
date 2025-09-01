import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Edge case to make sure kernel size with certain condition are with input size  (Rule 79)

rule_79 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_values"], 0) == 1, And(Select(v["arg2_shape"], 2) >= 1, If(Select(v["arg1_values"], 1) == 1, Select(v["arg2_shape"], 3) >= 1, True)), True)) if n else
          If(Select(v["arg1_values"], 0) == 1, And(Select(v["arg2_shape"], 2) >= 1, If(Select(v["arg1_values"], 1) == 1, Select(v["arg2_shape"], 3) >= 1, True)), True))
)

def rule_79_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 79
        rule_79(solver, {'arg1_values': arg1_values, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_79(solver, {'arg1_values': arg1['values'], 'arg2_shape': arg2['shape']}, neg)
