import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If size_average or reduce are specified, they must be boolean (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False)))) if n else
          And((Or(v["arg1_value"] == True, v["arg1_value"] == False)), (Or(v["arg2_value"] == True, v["arg2_value"] == False))))
)

def rule_12_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, bool) or isinstance(arg2, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
