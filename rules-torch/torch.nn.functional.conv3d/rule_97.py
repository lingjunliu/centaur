import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When the weight is an even length and dilation is odd, the performance is lowered so give a warning (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(If(And(And(((Select(v["arg1_shape"], 2) % 2) == 0), ((Select(v["arg1_shape"], 3) % 2) == 0)), ((Select(v["arg1_shape"], 4) % 2) == 0)), If(And(And(((Select(v["arg2_values"], 0) % 2) == 1), ((Select(v["arg2_values"], 1) % 2) == 1)), ((Select(v["arg2_values"], 2) % 2) == 1)), False, True), True)) if n else
          If(And(And(((Select(v["arg1_shape"], 2) % 2) == 0), ((Select(v["arg1_shape"], 3) % 2) == 0)), ((Select(v["arg1_shape"], 4) % 2) == 0)), If(And(And(((Select(v["arg2_values"], 0) % 2) == 1), ((Select(v["arg2_values"], 1) % 2) == 1)), ((Select(v["arg2_values"], 2) % 2) == 1)), False, True), True))
)

def rule_97_func(arg1, arg2, solver=None, neg=False):
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
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 97
        rule_97(solver, {'arg1_shape': arg1_shape, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_shape': arg1['shape'], 'arg2_values': arg2['values']}, neg)
