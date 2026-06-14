import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When data_format is NCHW, then ksize[0] and strides[0] should be 1 (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 32, And(Select(v["arg2_values"], 0) == 1, Select(v["arg3_values"], 0) == 1), True)) if n else
          If(v["arg1_value"] == 32, And(Select(v["arg2_values"], 0) == 1, Select(v["arg3_values"], 0) == 1), True))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values']}, neg)
