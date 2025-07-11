import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# One padding list should be all zeros if the other one is within a threshold (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(And(Select(v["arg1_values"], 0) <= v["arg3_value"], Select(v["arg1_values"], 1) <= v["arg3_value"]), And(Select(v["arg2_values"], 0) == 0, Select(v["arg2_values"], 1) == 0), False)) if n else
          If(And(Select(v["arg1_values"], 0) <= v["arg3_value"], Select(v["arg1_values"], 1) <= v["arg3_value"]), And(Select(v["arg2_values"], 0) == 0, Select(v["arg2_values"], 1) == 0), False))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 29
        rule_29(solver, {'arg1_values': arg1_values, 'arg2_values': arg2_values, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_values': arg1['values'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value']}, neg)
