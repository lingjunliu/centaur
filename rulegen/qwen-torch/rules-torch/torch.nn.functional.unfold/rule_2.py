import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# kernel size should be positive (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(And(And(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 1) > 0), v["arg2_value"] > 0)) if n else
          And(And(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 1) > 0), v["arg2_value"] > 0))
)

def rule_2_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 2
        rule_2(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)
