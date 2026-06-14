import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If data_format is channels_first, the channel dimension of input tensor should match with ksize and strides. (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 25, And((If(v["arg3_length"] > 1, Select(v["arg1_shape"], 1) == Select(v["arg3_values"], 0), True)), (If(v["arg4_length"] > 1, Select(v["arg1_shape"], 1) == Select(v["arg4_values"], 0), True))), True)) if n else
          If(v["arg2_value"] == 25, And((If(v["arg3_length"] > 1, Select(v["arg1_shape"], 1) == Select(v["arg3_values"], 0), True)), (If(v["arg4_length"] > 1, Select(v["arg1_shape"], 1) == Select(v["arg4_values"], 0), True))), True))
)

def rule_20_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg4_values = Array('arg4_values', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])
        solver.add(arg4_length == len(arg4))
        for i in range(len(arg4)):
            arg4_values = Store(arg4_values, i, arg4[i])

        # Constraints for rule 20
        rule_20(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_values': arg3_values, 'arg3_length': arg3_length, 'arg4_values': arg4_values, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length'], 'arg4_values': arg4['values'], 'arg4_length': arg4['length']}, neg)
