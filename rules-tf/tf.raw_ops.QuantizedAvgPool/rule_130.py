import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Height and width ksize values must be less than or equal to input height and width when padding is valid and that batch size channel must be 1 (Rule 130)

rule_130 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == 28, (And(And(And(Select(v["arg2_values"], 1) <= Select(v["arg1_shape"], 1), Select(v["arg2_values"], 2) <= Select(v["arg1_shape"], 2)), Select(v["arg4_values"], 0) == 1), Select(v["arg4_values"], 3) == 1)), True)) if n else
          If(v["arg3_value"] == 28, (And(And(And(Select(v["arg2_values"], 1) <= Select(v["arg1_shape"], 1), Select(v["arg2_values"], 2) <= Select(v["arg1_shape"], 2)), Select(v["arg4_values"], 0) == 1), Select(v["arg4_values"], 3) == 1)), True))
)

def rule_130_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 130
        rule_130(solver, {'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_values': arg4_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_130(solver, {'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_values': arg4['values']}, neg)
