import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if padding is VALID, stride = 1 in height and width dimensions. Also, the Ksize batch and channel must be 1 (Rule 140)

rule_140 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 28, And(And(And(Select(v["arg1_values"], 1) == 1, Select(v["arg1_values"], 2) == 1), Select(v["arg3_values"], 0) == 1), Select(v["arg3_values"], 3) == 1), True)) if n else
          If(v["arg2_value"] == 28, And(And(And(Select(v["arg1_values"], 1) == 1, Select(v["arg1_values"], 2) == 1), Select(v["arg3_values"], 0) == 1), Select(v["arg3_values"], 3) == 1), True))
)

def rule_140_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 140
        rule_140(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_140(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values']}, neg)
