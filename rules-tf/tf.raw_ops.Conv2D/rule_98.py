import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# strides values layout dependency on data format (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_length"] == 4, (Or((And(And(v["arg1_value"] == 31, Select(v["arg2_values"], 0) == 1), Select(v["arg2_values"], 3) == 1)), (And(And(v["arg1_value"] == 32, Select(v["arg2_values"], 0) == 1), Select(v["arg2_values"], 1) == 1)))))) if n else
          And(v["arg2_length"] == 4, (Or((And(And(v["arg1_value"] == 31, Select(v["arg2_values"], 0) == 1), Select(v["arg2_values"], 3) == 1)), (And(And(v["arg1_value"] == 32, Select(v["arg2_values"], 0) == 1), Select(v["arg2_values"], 1) == 1))))))
)

def rule_98_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 98
        rule_98(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length']}, neg)
