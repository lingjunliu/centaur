import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if centered is true and normalized is false, offsets should be less than height and width of the image taking into account the size. (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"], (v["arg4_value"] == False)), (And(Select(v["arg2_range"], 1) + Select(v["arg5_shape"], 1) / 2 <= Select(v["arg1_shape"], 1), Select(v["arg2_range"], 1) + Select(v["arg5_shape"], 0) / 2 <= Select(v["arg1_shape"], 2))), True)) if n else
          If(And(v["arg3_value"], (v["arg4_value"] == False)), (And(Select(v["arg2_range"], 1) + Select(v["arg5_shape"], 1) / 2 <= Select(v["arg1_shape"], 1), Select(v["arg2_range"], 1) + Select(v["arg5_shape"], 0) / 2 <= Select(v["arg1_shape"], 2))), True))
)

def rule_16_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_shape = Array('arg5_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        for i in range(arg5.ndim):
            arg5_shape = Store(arg5_shape, i, arg5.shape[i])

        # Constraints for rule 16
        rule_16(solver, {'arg1_shape': arg1_shape, 'arg2_range': arg2_range, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_shape': arg5_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_shape': arg1['shape'], 'arg2_range': arg2['range'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_shape': arg5['shape']}, neg)
