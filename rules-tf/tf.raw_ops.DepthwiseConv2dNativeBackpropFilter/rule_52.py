import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When padding is explicit and data format is NCHW, the explicit paddings values should be less than or equal to the input height and width plus filter height and width. (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == 30, v["arg5_value"] == 34), And(And(And(Select(v["arg2_values"], 0) <= Select(v["arg3_shape"], 2) + Select(v["arg4_shape"], 0), Select(v["arg2_values"], 1) <= Select(v["arg3_shape"], 2) + Select(v["arg4_shape"], 0)), Select(v["arg2_values"], 2) <= Select(v["arg3_shape"], 3) + Select(v["arg4_shape"], 1)), Select(v["arg2_values"], 3) <= Select(v["arg3_shape"], 3) + Select(v["arg4_shape"], 1)), True)) if n else
          If(And(v["arg1_value"] == 30, v["arg5_value"] == 34), And(And(And(Select(v["arg2_values"], 0) <= Select(v["arg3_shape"], 2) + Select(v["arg4_shape"], 0), Select(v["arg2_values"], 1) <= Select(v["arg3_shape"], 2) + Select(v["arg4_shape"], 0)), Select(v["arg2_values"], 2) <= Select(v["arg3_shape"], 3) + Select(v["arg4_shape"], 1)), Select(v["arg2_values"], 3) <= Select(v["arg3_shape"], 3) + Select(v["arg4_shape"], 1)), True))
)

def rule_52_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_value == list_of_string_values_tf.index(arg5))

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_shape': arg3_shape, 'arg4_shape': arg4_shape, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_shape': arg3['shape'], 'arg4_shape': arg4['shape'], 'arg5_value': arg5['value']}, neg)
