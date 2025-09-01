import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If any of the string attributes (name, container, shared_name (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 6, v["arg2_value"] == 6), v["arg3_value"] == 6), v["arg4_length"] == 0, True)) if n else
          If(Or(Or(v["arg1_value"] == 6, v["arg2_value"] == 6), v["arg3_value"] == 6), v["arg4_length"] == 0, True))
)

def rule_99_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = String('arg3_value')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 99
        rule_99(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_length': arg4['length']}, neg)
