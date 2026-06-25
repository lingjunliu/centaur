import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data format is NCHW, the height and width dimensions of input must be greater or equal than ksize and strides height and width respectively (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 34, And(And(And(Select(v["arg4_shape"], 2) >= Select(v["arg2_values"], 2), Select(v["arg4_shape"], 3) >= Select(v["arg2_values"], 3)), Select(v["arg4_shape"], 2) >= Select(v["arg3_values"], 2)), Select(v["arg4_shape"], 3) >= Select(v["arg3_values"], 3)), True)) if n else
          If(v["arg1_value"] == 34, And(And(And(Select(v["arg4_shape"], 2) >= Select(v["arg2_values"], 2), Select(v["arg4_shape"], 3) >= Select(v["arg2_values"], 3)), Select(v["arg4_shape"], 2) >= Select(v["arg3_values"], 2)), Select(v["arg4_shape"], 3) >= Select(v["arg3_values"], 3)), True))
)

def rule_18_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])

        # Constraints for rule 18
        rule_18(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values, 'arg3_values': arg3_values, 'arg4_shape': arg4_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values'], 'arg3_values': arg3['values'], 'arg4_shape': arg4['shape']}, neg)
