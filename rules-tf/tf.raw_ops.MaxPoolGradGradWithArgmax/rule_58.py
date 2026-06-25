import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if include_batch_in_index is false, shape of argmax should be [height, width, channel] (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, v["arg4_value"] == 29), And(Select(v["arg2_shape"], 1) == ((Select(v["arg3_shape"], 1) + Select(v["arg5_values"], 1) - 1) / Select(v["arg5_values"], 1)), Select(v["arg2_shape"], 2) == ((Select(v["arg3_shape"], 2) + Select(v["arg5_values"], 2) - 1) / Select(v["arg5_values"], 2))), If(And(v["arg1_value"] == False, v["arg4_value"] == 28), And(Select(v["arg2_shape"], 1) == (Select(v["arg3_shape"], 1) - (Select(v["arg5_values"], 1) - 1) - 1) / Select(v["arg5_values"], 1) + 1, Select(v["arg2_shape"], 2) == (Select(v["arg3_shape"], 2) - (Select(v["arg5_values"], 2) - 1) - 1) / Select(v["arg5_values"], 2) + 1), True))) if n else
          If(And(v["arg1_value"] == False, v["arg4_value"] == 29), And(Select(v["arg2_shape"], 1) == ((Select(v["arg3_shape"], 1) + Select(v["arg5_values"], 1) - 1) / Select(v["arg5_values"], 1)), Select(v["arg2_shape"], 2) == ((Select(v["arg3_shape"], 2) + Select(v["arg5_values"], 2) - 1) / Select(v["arg5_values"], 2))), If(And(v["arg1_value"] == False, v["arg4_value"] == 28), And(Select(v["arg2_shape"], 1) == (Select(v["arg3_shape"], 1) - (Select(v["arg5_values"], 1) - 1) - 1) / Select(v["arg5_values"], 1) + 1, Select(v["arg2_shape"], 2) == (Select(v["arg3_shape"], 2) - (Select(v["arg5_values"], 2) - 1) - 1) / Select(v["arg5_values"], 2) + 1), True)))
)

def rule_58_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_value = Int('arg4_value')
        arg5_values = Array('arg5_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_value == list_of_string_values_tf.index(arg4))
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_value': arg4_value, 'arg5_values': arg5_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_value': arg4['value'], 'arg5_values': arg5['values']}, neg)
