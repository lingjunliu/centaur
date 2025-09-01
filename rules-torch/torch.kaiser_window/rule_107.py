import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If window_length is less than 2 and periodic is true, then beta must be zero and requires_grad must be false and dtype must support this (Rule 107)

rule_107 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] < 2, v["arg3_value"]), And(And(v["arg2_value"] == 0, v["arg4_value"] == False), (Or(Or(Or(Or(Or(v["arg5_value"] == 6, v["arg5_value"] == 7), v["arg5_value"] == 8), v["arg5_value"] == 9), v["arg5_value"] == 10), v["arg5_value"] < 6))), True)) if n else
          If(And(v["arg1_value"] < 2, v["arg3_value"]), And(And(v["arg2_value"] == 0, v["arg4_value"] == False), (Or(Or(Or(Or(Or(v["arg5_value"] == 6, v["arg5_value"] == 7), v["arg5_value"] == 8), v["arg5_value"] == 9), v["arg5_value"] == 10), v["arg5_value"] < 6))), True))
)

def rule_107_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not (isinstance(arg5, torch.dtype) or isinstance(arg5, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Int('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == list_of_available_dtypes.index(np_dtype(arg5)))

        # Constraints for rule 107
        rule_107(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
