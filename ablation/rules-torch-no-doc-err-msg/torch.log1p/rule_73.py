import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The sum of elements in a tuple must be less than the minimum value of a tensor (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] > 0, And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg2_range"], 0) > Select(v["arg1_values"], i)) for i in range(6)]))) if n else
          And(v["arg1_length"] > 0, And([Implies(i < (v["arg1_length"] - 1 + 1), Select(v["arg2_range"], 0) > Select(v["arg1_values"], i)) for i in range(6)])))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
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
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 73
        rule_73(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_range': arg2['range']}, neg)
