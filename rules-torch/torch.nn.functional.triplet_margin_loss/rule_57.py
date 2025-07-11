import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Reduction must not be constant, swap must be a boolean value, and p must be non-negative (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] != 20, (Or(v["arg2_value"] == True, v["arg2_value"] == False))), v["arg3_value"] >= 0)) if n else
          And(And(v["arg1_value"] != 20, (Or(v["arg2_value"] == True, v["arg2_value"] == False))), v["arg3_value"] >= 0))
)

def rule_57_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 57
        rule_57(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
