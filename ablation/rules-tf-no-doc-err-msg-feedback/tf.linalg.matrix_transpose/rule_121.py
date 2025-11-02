import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the perms are given, and the input tensor is a matrix, then its elements must be 0 and 1. (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 2, v["arg2_length"] == 2), Or((And(Select(v["arg2_values"], 0) == 0, Select(v["arg2_values"], 1) == 1)), (And(Select(v["arg2_values"], 0) == 1, Select(v["arg2_values"], 1) == 0))), True)) if n else
          If(And(v["arg1_ndim"] == 2, v["arg2_length"] == 2), Or((And(Select(v["arg2_values"], 0) == 0, Select(v["arg2_values"], 1) == 1)), (And(Select(v["arg2_values"], 0) == 1, Select(v["arg2_values"], 1) == 0))), True))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 121
        rule_121(solver, {'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
