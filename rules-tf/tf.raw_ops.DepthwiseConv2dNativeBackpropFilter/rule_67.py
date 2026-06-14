import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When padding is EXPLICIT then strides.len, dilations.len must be equal to input.ndim AND explicit_paddings.len must be even. (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 28, And(And(And(And(v["arg4_length"] == v["arg2_ndim"], v["arg5_length"] == v["arg2_ndim"]), v["arg6_length"] % 2 == 0), Select(v["arg3_shape"], 0) > 0), Select(v["arg3_shape"], 1) > 0), True)) if n else
          If(v["arg1_value"] == 28, And(And(And(And(v["arg4_length"] == v["arg2_ndim"], v["arg5_length"] == v["arg2_ndim"]), v["arg6_length"] % 2 == 0), Select(v["arg3_shape"], 0) > 0), Select(v["arg3_shape"], 1) > 0), True))
)

def rule_67_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False
        if not (isinstance(arg6, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg6)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())
        arg4_length = Int('arg4_length')
        arg5_length = Int('arg5_length')
        arg6_length = Int('arg6_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])
        solver.add(arg4_length == len(arg4))
        solver.add(arg5_length == len(arg5))
        solver.add(arg6_length == len(arg6))

        # Constraints for rule 67
        rule_67(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape, 'arg4_length': arg4_length, 'arg5_length': arg5_length, 'arg6_length': arg6_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape'], 'arg4_length': arg4['length'], 'arg5_length': arg5['length'], 'arg6_length': arg6['length']}, neg)
