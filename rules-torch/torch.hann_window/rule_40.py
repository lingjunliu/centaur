import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Ensure that numy.int16 isn't allowed as well as window_length should be positive (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 2, False, And(And(And(And((Or(v["arg2_value"] == True, v["arg2_value"] == False)), (v["arg3_value"] == 6)), (Or(Or(v["arg4_value"] == 0, v["arg4_value"] == 7), v["arg4_value"] == 8))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), v["arg1_value"] >= 0))) if n else
          If(v["arg1_value"] == 2, False, And(And(And(And((Or(v["arg2_value"] == True, v["arg2_value"] == False)), (v["arg3_value"] == 6)), (Or(Or(v["arg4_value"] == 0, v["arg4_value"] == 7), v["arg4_value"] == 8))), (Or(v["arg5_value"] == True, v["arg5_value"] == False))), v["arg1_value"] >= 0)))
)

def rule_40_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False
        if not isinstance(arg5, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = String('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_value = Bool('arg5_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))
        solver.add(arg5_value == arg5)

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value']}, neg)
