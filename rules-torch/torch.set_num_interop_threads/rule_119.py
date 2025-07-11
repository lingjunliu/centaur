import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check value only when very flag is met (Rule 119)

rule_119 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(v["arg6_length"] > 0, v["arg5_length"] == 0), v["arg2_value"] == 11), v["arg4_value"] == 1), v["arg3_value"]), v["arg1_value"] == 20, False)) if n else
          If(And(And(And(And(v["arg6_length"] > 0, v["arg5_length"] == 0), v["arg2_value"] == 11), v["arg4_value"] == 1), v["arg3_value"]), v["arg1_value"] == 20, False))
)

def rule_119_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, bool):
            return False
        if not (isinstance(arg4, torch.dtype) or isinstance(arg4, tf.dtypes.DType)):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False
        if not (isinstance(arg6, tuple) and all(isinstance(e, (float, np.floating)) for e in arg6)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Int('arg4_value')
        arg5_length = Int('arg5_length')
        arg6_length = Int('arg6_length')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == list_of_available_dtypes.index(np_dtype(arg4)))
        solver.add(arg5_length == len(arg5))
        solver.add(arg6_length == len(arg6))

        # Constraints for rule 119
        rule_119(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_length': arg5_length, 'arg6_length': arg6_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_length': arg5['length'], 'arg6_length': arg6['length']}, neg)
