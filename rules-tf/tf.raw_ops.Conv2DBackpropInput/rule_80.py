import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if explicit padding, then the length of explicit paddings must be twice the number of spatial dimensions (Rule 80)

rule_80 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 28, v["arg3_value"] == 31), v["arg2_length"] == 2 * 2, If(And(v["arg1_value"] == 28, v["arg3_value"] == 32), v["arg2_length"] == 2 * 2, True))) if n else
          If(And(v["arg1_value"] == 28, v["arg3_value"] == 31), v["arg2_length"] == 2 * 2, If(And(v["arg1_value"] == 28, v["arg3_value"] == 32), v["arg2_length"] == 2 * 2, True)))
)

def rule_80_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 80
        rule_80(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_80(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
