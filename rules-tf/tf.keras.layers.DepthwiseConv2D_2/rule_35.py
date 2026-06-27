import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# effective height constraint for channels_last with valid padding (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == 21, v["arg4_value"] == 24), Select(v["arg1_shape"], 1) >= (Select(v["arg2_values"], 0) - 1) * Select(v["arg5_values"], 0) + 1, True)) if n else
          If(And(v["arg3_value"] == 21, v["arg4_value"] == 24), Select(v["arg1_shape"], 1) >= (Select(v["arg2_values"], 0) - 1) * Select(v["arg5_values"], 0) + 1, True))
)

def rule_35_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, str):
            return False
        if not isinstance(arg4, str):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_value = String('arg3_value')
        arg4_value = String('arg4_value')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 35
        rule_35(solver, {'arg1_shape': arg1_shape, 'arg2_values': arg2_values, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_shape': arg1['shape'], 'arg2_values': arg2['values'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_values': arg5['values']}, neg)
