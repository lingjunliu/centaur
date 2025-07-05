import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# List needs to follow these conditions. (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(And((Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 0) for i in range(6)])), v["arg1_value"] < 10)) if n else
          And((Or([And(i < (v["arg2_length"] - 1 + 1), Select(v["arg2_values"], i) > 0) for i in range(6)])), v["arg1_value"] < 10))
)

def rule_93_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 93
        rule_93(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
