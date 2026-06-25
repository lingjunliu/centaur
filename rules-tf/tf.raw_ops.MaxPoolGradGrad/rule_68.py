import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the data format is "NCHW", then the strides dimensions must be smaller or equal than the dimensions of the input tensor, excluding the batch and channel dimensions if padding is VALID (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 34, v["arg4_value"] == 28), And(Select(v["arg3_values"], 2) <= Select(v["arg2_shape"], 2), Select(v["arg3_values"], 3) <= Select(v["arg2_shape"], 3)), True)) if n else
          If(And(v["arg1_value"] == 34, v["arg4_value"] == 28), And(Select(v["arg3_values"], 2) <= Select(v["arg2_shape"], 2), Select(v["arg3_values"], 3) <= Select(v["arg2_shape"], 3)), True))
)

def rule_68_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 68
        rule_68(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value']}, neg)
