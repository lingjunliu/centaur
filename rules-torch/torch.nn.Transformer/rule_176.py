import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Enforce feature dimension compatibility based on batch_first when not using custom encoders and decoders (Rule 176)

rule_176 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] == 6, v["arg5_value"] == 6), (If(v["arg6_value"] == False, (And(Select(v["arg1_shape"], 2) == v["arg3_value"], Select(v["arg2_shape"], 2) == v["arg3_value"])), (And(Select(v["arg1_shape"], 1) == v["arg3_value"], Select(v["arg2_shape"], 1) == v["arg3_value"])))), False)) if n else
          If(And(v["arg4_value"] == 6, v["arg5_value"] == 6), (If(v["arg6_value"] == False, (And(Select(v["arg1_shape"], 2) == v["arg3_value"], Select(v["arg2_shape"], 2) == v["arg3_value"])), (And(Select(v["arg1_shape"], 1) == v["arg3_value"], Select(v["arg2_shape"], 1) == v["arg3_value"])))), False))
)

def rule_176_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, str):
            return False
        if not isinstance(arg5, str):
            return False
        if not isinstance(arg6, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_value = Int('arg3_value')
        arg4_value = String('arg4_value')
        arg5_value = String('arg5_value')
        arg6_value = Bool('arg6_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == list_of_string_values_torch.index(arg4))
        solver.add(arg5_value == list_of_string_values_torch.index(arg5))
        solver.add(arg6_value == arg6)

        # Constraints for rule 176
        rule_176(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_176(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
