import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Combined check for theta shape and size length (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg2_shape"], 1) == 2, Select(v["arg2_shape"], 2) == 3), v["arg1_length"] == 4, If(And(Select(v["arg2_shape"], 1) == 3, Select(v["arg2_shape"], 2) == 4), v["arg1_length"] == 5, True))) if n else
          If(And(Select(v["arg2_shape"], 1) == 2, Select(v["arg2_shape"], 2) == 3), v["arg1_length"] == 4, If(And(Select(v["arg2_shape"], 1) == 3, Select(v["arg2_shape"], 2) == 4), v["arg1_length"] == 5, True)))
)

def rule_59_func(arg1, arg2, solver=None, neg=False):
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
        arg1_length = Int('arg1_length')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 59
        rule_59(solver, {'arg1_length': arg1_length, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_length': arg1['length'], 'arg2_shape': arg2['shape']}, neg)
