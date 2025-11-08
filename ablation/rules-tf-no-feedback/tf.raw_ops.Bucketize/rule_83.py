import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When dtype is int, boundary must have int representable values (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3), And([Implies(i < (If(v["arg2_length"] > 0, v["arg2_length"] - 1, 0) + 1), Select(v["arg2_values"], i) % 1 == 0) for i in range(6)]), True)) if n else
          If(Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 3), And([Implies(i < (If(v["arg2_length"] > 0, v["arg2_length"] - 1, 0) + 1), Select(v["arg2_values"], i) % 1 == 0) for i in range(6)]), True))
)

def rule_83_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 83
        rule_83(solver, {'arg1_dtype': arg1_dtype, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_dtype': arg1['dtype'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
