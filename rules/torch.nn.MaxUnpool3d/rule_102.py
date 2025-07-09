import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If output_size's len is 5 tensor dimensions needs to be within range (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_length"] == 5, And(And(And(And(And(Select(v["arg1_shape"], 2) > 0, Select(v["arg1_shape"], 2) < 1000), Select(v["arg1_shape"], 3) > 0), Select(v["arg1_shape"], 3) < 1000), Select(v["arg1_shape"], 4) > 0), Select(v["arg1_shape"], 4) < 1000), False)) if n else
          If(v["arg2_length"] == 5, And(And(And(And(And(Select(v["arg1_shape"], 2) > 0, Select(v["arg1_shape"], 2) < 1000), Select(v["arg1_shape"], 3) > 0), Select(v["arg1_shape"], 3) < 1000), Select(v["arg1_shape"], 4) > 0), Select(v["arg1_shape"], 4) < 1000), False))
)

def rule_102_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 102
        rule_102(solver, {'arg1_shape': arg1_shape, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_shape': arg1['shape'], 'arg2_length': arg2['length']}, neg)
