import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Dummy Rule: If v_1 is a tuple of integers with length 2, and v_2 is a string, then v_2 cannot be equal to a specific string (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 2, v["arg2_value"] != 0, False)) if n else
          If(v["arg1_length"] == 2, v["arg2_value"] != 0, False))
)

def rule_29_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))

        # Constraints for rule 29
        rule_29(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
