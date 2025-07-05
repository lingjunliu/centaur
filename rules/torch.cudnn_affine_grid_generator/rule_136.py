import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Check the length for outputSize based on theta (Rule 136)

rule_136 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 1) == 2, v["arg2_length"] == 4, If(Select(v["arg1_shape"], 1) == 3, v["arg2_length"] == 5, False))) if n else
          If(Select(v["arg1_shape"], 1) == 2, v["arg2_length"] == 4, If(Select(v["arg1_shape"], 1) == 3, v["arg2_length"] == 5, False)))
)

def rule_136_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 136
        rule_136(solver, {'arg1_shape': arg1_shape, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_136(solver, {'arg1_shape': arg1['shape'], 'arg2_length': arg2['length']}, neg)
