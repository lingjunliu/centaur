import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel size must be positive (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > 0, Select(v["arg2_values"], 0) > 0), Select(v["arg2_values"], 1) > 0)) if n else
          And(And(v["arg1_value"] > 0, Select(v["arg2_values"], 0) > 0), Select(v["arg2_values"], 1) > 0))
)

def rule_1_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 1
        rule_1(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values']}, neg)
