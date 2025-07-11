import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Depend on List+ String and Tuple to make this process occur (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_length"] > 0, v["arg3_value"] == 9), v["arg1_value"] > v["arg4_length"], False)) if n else
          If(And(v["arg2_length"] > 0, v["arg3_value"] == 9), v["arg1_value"] > v["arg4_length"], False))
)

def rule_111_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, str):
            return False
        if not (isinstance(arg4, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_value = String('arg3_value')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 111
        rule_111(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_length': arg4['length']}, neg)
