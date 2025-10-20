import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If size_average is None and reduce is not None, then reduction should default to 'mean' (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), v["arg2_value"] == False), v["arg3_value"] == 7, True)) if n else
          If(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), v["arg2_value"] == False), v["arg3_value"] == 7, True))
)

def rule_29_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 29
        rule_29(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
