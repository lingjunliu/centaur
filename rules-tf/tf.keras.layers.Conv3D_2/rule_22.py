import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input spatial dimensions must support the dilated kernel size under channels_last format (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg5_ndim"] == 5, v["arg1_length"] == 3), v["arg4_length"] == 3), (If(And(v["arg2_value"] == 21, v["arg3_value"] == 24), (And([Implies(i < (3 + 1), Select(v["arg5_shape"], i) >= (Select(v["arg1_values"], i - 1) - 1) * Select(v["arg4_values"], i - 1) + 1) for i in range(6)])), True)))) if n else
          And(And(And(v["arg5_ndim"] == 5, v["arg1_length"] == 3), v["arg4_length"] == 3), (If(And(v["arg2_value"] == 21, v["arg3_value"] == 24), (And([Implies(i < (3 + 1), Select(v["arg5_shape"], i) >= (Select(v["arg1_values"], i - 1) - 1) * Select(v["arg4_values"], i - 1) + 1) for i in range(6)])), True))))
)

def rule_22_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = String('arg2_value')
        arg3_value = String('arg3_value')
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())
        arg5_ndim = Int('arg5_ndim')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])
        solver.add(arg5_ndim == arg5.ndim)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 22
        rule_22(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_values': arg4_values, 'arg4_length': arg4_length, 'arg5_ndim': arg5_ndim, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length'], 'arg5_ndim': arg5['ndim'], 'arg5_shape': arg5['shape']}, neg)
