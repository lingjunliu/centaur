import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# padding, data format, and strides length constraints (Rule 130)

rule_130 = lambda s, v, n=False: (
    s.add(Not(And(And((Or(v["arg1_value"] == 22, v["arg1_value"] == 21)), (Or(v["arg2_value"] == 24, v["arg2_value"] == 25))), v["arg3_length"] == 5)) if n else
          And(And((Or(v["arg1_value"] == 22, v["arg1_value"] == 21)), (Or(v["arg2_value"] == 24, v["arg2_value"] == 25))), v["arg3_length"] == 5))
)

def rule_130_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 130
        rule_130(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_130(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
