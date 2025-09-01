import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check the hidden state shape matches the requirements when batch_first is True (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] == True, v["arg5_value"] == True), And(And(Select(v["arg1_shape"], 0) == 2 * v["arg2_value"], Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) == v["arg3_value"]), If(And(v["arg4_value"] == True, v["arg5_value"] == False), And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) == v["arg3_value"]), True))) if n else
          If(And(v["arg4_value"] == True, v["arg5_value"] == True), And(And(Select(v["arg1_shape"], 0) == 2 * v["arg2_value"], Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) == v["arg3_value"]), If(And(v["arg4_value"] == True, v["arg5_value"] == False), And(And(Select(v["arg1_shape"], 0) == v["arg2_value"], Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) == v["arg3_value"]), True)))
)

def rule_56_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)

        # Constraints for rule 56
        rule_56(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
