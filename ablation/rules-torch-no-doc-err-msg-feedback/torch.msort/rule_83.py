import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input v_1 is nearest then input v_2 must be an empty tuple. (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 25, v["arg2_length"] == 0, True)) if n else
          If(v["arg1_value"] == 25, v["arg2_length"] == 0, True))
)

def rule_83_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 83
        rule_83(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
