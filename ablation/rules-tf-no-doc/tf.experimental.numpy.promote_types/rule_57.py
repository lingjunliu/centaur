import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Type promotion between a tuple of floats and an int, if int is less than 10, the shape should be larger than  (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_shape"], 0) > 10, v["arg1_length"] < 5, v["arg1_length"] > 5)) if n else
          If(Select(v["arg2_shape"], 0) > 10, v["arg1_length"] < 5, v["arg1_length"] > 5))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
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

        # Constraints for rule 57
        rule_57(solver, {'arg1_length': arg1_length, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_length': arg1['length'], 'arg2_shape': arg2['shape']}, neg)
