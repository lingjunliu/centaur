import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# At least one of int8, int16, int32, int64 or float dtypes is defined if a memory limit is given. (Rule 116)

rule_116 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] > 0, Or([And(i < (v["arg2_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Select(v["arg2_values"], i) == 1, Select(v["arg2_values"], i) == 2), Select(v["arg2_values"], i) == 3), Select(v["arg2_values"], i) == 4), Select(v["arg2_values"], i) == 6), Select(v["arg2_values"], i) == 7), Select(v["arg2_values"], i) == 8)) for i in range(6)]), True)) if n else
          If(v["arg1_value"] > 0, Or([And(i < (v["arg2_length"] - 1 + 1), Or(Or(Or(Or(Or(Or(Select(v["arg2_values"], i) == 1, Select(v["arg2_values"], i) == 2), Select(v["arg2_values"], i) == 3), Select(v["arg2_values"], i) == 4), Select(v["arg2_values"], i) == 6), Select(v["arg2_values"], i) == 7), Select(v["arg2_values"], i) == 8)) for i in range(6)]), True))
)

def rule_116_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 116
        rule_116(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_116(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
