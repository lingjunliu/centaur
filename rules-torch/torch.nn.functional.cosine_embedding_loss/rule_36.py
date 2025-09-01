import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If reduction is not 'none', and reduce is specified then the type of reduce should be bool (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] != 6, v["arg1_value"] != 6), Or(v["arg1_value"] == True, v["arg1_value"] == False), True)) if n else
          If(And(v["arg2_value"] != 6, v["arg1_value"] != 6), Or(v["arg1_value"] == True, v["arg1_value"] == False), True))
)

def rule_36_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or isinstance(arg1, str)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
