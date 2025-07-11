import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# a boolean and a string must equal 'true' and 'none' respectively  (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] == True, v["arg2_value"] == 6)) if n else
          And(v["arg1_value"] == True, v["arg2_value"] == 6))
)

def rule_93_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))

        # Constraints for rule 93
        rule_93(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
