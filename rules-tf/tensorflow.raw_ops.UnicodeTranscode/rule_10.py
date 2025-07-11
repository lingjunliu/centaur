import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input_encoding and output_encoding cannot be the same when replace_control_characters is true (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == True, v["arg1_value"] != v["arg2_value"], False)) if n else
          If(v["arg3_value"] == True, v["arg1_value"] != v["arg2_value"], False))
)

def rule_10_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == arg3)

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
