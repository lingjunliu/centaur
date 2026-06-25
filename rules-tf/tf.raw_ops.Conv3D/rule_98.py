import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If using VALID padding, check the relation between the input size, filter size and strides values (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == 28, And(And(Select(v["arg1_shape"], 1) >= (Select(v["arg2_shape"], 0) - 1) * Select(v["arg3_values"], 1), Select(v["arg1_shape"], 2) >= (Select(v["arg2_shape"], 1) - 1) * Select(v["arg3_values"], 2)), Select(v["arg1_shape"], 3) >= (Select(v["arg2_shape"], 2) - 1) * Select(v["arg3_values"], 3)), True)) if n else
          If(v["arg4_value"] == 28, And(And(Select(v["arg1_shape"], 1) >= (Select(v["arg2_shape"], 0) - 1) * Select(v["arg3_values"], 1), Select(v["arg1_shape"], 2) >= (Select(v["arg2_shape"], 1) - 1) * Select(v["arg3_values"], 2)), Select(v["arg1_shape"], 3) >= (Select(v["arg2_shape"], 2) - 1) * Select(v["arg3_values"], 3)), True))
)

def rule_98_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_value = Int('arg4_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))

        # Constraints for rule 98
        rule_98(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_values': arg3_values, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_values': arg3['values'], 'arg4_value': arg4['value']}, neg)
