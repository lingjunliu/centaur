import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Make sure that the length is 1 or 2 of output_size, and if it is one, then the sizes of the H and W dimensions of the tensor are at least greater than 0 (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 1, And(Select(v["arg2_shape"], 2) > 0, Select(v["arg2_shape"], 3) > 0), If(v["arg1_length"] == 2, True, v["arg1_length"] == 0))) if n else
          If(v["arg1_length"] == 1, And(Select(v["arg2_shape"], 2) > 0, Select(v["arg2_shape"], 3) > 0), If(v["arg1_length"] == 2, True, v["arg1_length"] == 0)))
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
