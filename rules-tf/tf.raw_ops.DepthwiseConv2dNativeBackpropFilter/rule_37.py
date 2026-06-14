import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If explicit padding is used, then strides and dilations must have length equal to the number of spatial dimensions plus 2 (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 28, If(v["arg2_value"] == 31, And(v["arg3_length"] == 4, v["arg4_length"] == 4), If(v["arg2_value"] == 32, And(v["arg3_length"] == 4, v["arg4_length"] == 4), True)), True)) if n else
          If(v["arg1_value"] == 28, If(v["arg2_value"] == 31, And(v["arg3_length"] == 4, v["arg4_length"] == 4), If(v["arg2_value"] == 32, And(v["arg3_length"] == 4, v["arg4_length"] == 4), True)), True))
)

def rule_37_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 37
        rule_37(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_length': arg3_length, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length'], 'arg4_length': arg4['length']}, neg)
