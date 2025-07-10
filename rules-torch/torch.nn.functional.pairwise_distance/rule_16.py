import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The first dimension of x1 and x2 must match, if not 1 (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_shape"], 0) != 1, Select(v["arg2_shape"], 0) != 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), False)) if n else
          If(And(Select(v["arg1_shape"], 0) != 1, Select(v["arg2_shape"], 0) != 1), Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0), False))
)

def rule_16_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 16
        rule_16(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape']}, neg)
