import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if include_batch_in_index is true, height and width of input and argmax should follow this equation when padding is VALID (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, v["arg6_value"] == 28), And(Select(v["arg2_shape"], 1) == (Select(v["arg3_shape"], 1) - (Select(v["arg4_values"], 1) - 1) - 1) / Select(v["arg5_values"], 1) + 1, Select(v["arg2_shape"], 2) == (Select(v["arg3_shape"], 2) - (Select(v["arg4_values"], 2) - 1) - 1) / Select(v["arg5_values"], 2) + 1), True)) if n else
          If(And(v["arg1_value"] == True, v["arg6_value"] == 28), And(Select(v["arg2_shape"], 1) == (Select(v["arg3_shape"], 1) - (Select(v["arg4_values"], 1) - 1) - 1) / Select(v["arg5_values"], 1) + 1, Select(v["arg2_shape"], 2) == (Select(v["arg3_shape"], 2) - (Select(v["arg4_values"], 2) - 1) - 1) / Select(v["arg5_values"], 2) + 1), True))
)

def rule_60_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False
        if not isinstance(arg6, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_values = Array('arg5_values', IntSort(), IntSort())
        arg6_value = Int('arg6_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        for i in range(len(arg5)):
            arg5_values = Store(arg5_values, i, arg5[i])
        solver.add(arg6_value == list_of_string_values_tf.index(arg6))

        # Constraints for rule 60
        rule_60(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape, 'arg4_values': arg4_values, 'arg5_values': arg5_values, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape'], 'arg4_values': arg4['values'], 'arg5_values': arg5['values'], 'arg6_value': arg6['value']}, neg)
