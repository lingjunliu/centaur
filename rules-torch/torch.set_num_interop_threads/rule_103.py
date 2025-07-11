import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# In very specific case when String = constant and certain DType is valid, our current integer will need to be the flag + some condition with List (Rule 103)

rule_103 = lambda s, v, n=False: (
    s.add(Not(If(v["arg5_length"] == 0, If(And(And(v["arg2_value"] == 10, v["arg3_value"] == 1), v["arg4_value"]), v["arg1_value"] == 10, False), False)) if n else
          If(v["arg5_length"] == 0, If(And(And(v["arg2_value"] == 10, v["arg3_value"] == 1), v["arg4_value"]), v["arg1_value"] == 10, False), False))
)

def rule_103_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False
        if not isinstance(arg4, bool):
            return False
        if not (isinstance(arg5, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg5)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_length = Int('arg5_length')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values_torch.torch.index(arg2))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))
        solver.add(arg4_value == arg4)
        solver.add(arg5_length == len(arg5))

        # Constraints for rule 103
        rule_103(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_length': arg5_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_103(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_length': arg5['length']}, neg)
