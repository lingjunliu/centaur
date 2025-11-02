import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The length of tuple v_1 should be greater than v_2 and the elements at indexes v_2 and v_2+1 must be positive (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_length"] > v["arg2_value"], Select(v["arg1_values"], v["arg2_value"]) > 0), Select(v["arg1_values"], v["arg2_value"] + 1) > 0)) if n else
          And(And(v["arg1_length"] > v["arg2_value"], Select(v["arg1_values"], v["arg2_value"]) > 0), Select(v["arg1_values"], v["arg2_value"] + 1) > 0))
)

def rule_99_func(arg1, arg2, solver=None, neg=False):
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
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 99
        rule_99(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
