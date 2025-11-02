import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When dilation is 1 and stride is 1, the output size equals the input size if padding is zero and ceil_mode is true (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(v["arg1_value"] == 1, v["arg2_value"] == 1), v["arg3_value"] == 0), v["arg5_value"] == True), Select(v["arg4_shape"], 2) == Select(v["arg4_shape"], 2), True)) if n else
          If(And(And(And(v["arg1_value"] == 1, v["arg2_value"] == 1), v["arg3_value"] == 0), v["arg5_value"] == True), Select(v["arg4_shape"], 2) == Select(v["arg4_shape"], 2), True))
)

def rule_72_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_shape = Array('arg4_shape', IntSort(), IntSort())
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        for i in range(arg4.ndim):
            arg4_shape = Store(arg4_shape, i, arg4.shape[i])
        solver.add(arg5_value == arg5)

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_shape': arg4_shape, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_shape': arg4['shape'], 'arg5_value': arg5['value']}, neg)
