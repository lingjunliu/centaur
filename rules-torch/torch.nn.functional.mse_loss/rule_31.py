import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Either reduce or reduction should be specified and if both are specified, their values should be consistent (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(Or(Or((v["arg1_value"] == none), (v["arg2_value"] == 6)), (Or((And(v["arg1_value"] == True, v["arg2_value"] == 7)), (And(v["arg1_value"] == False, v["arg2_value"] == 8)))))) if n else
          Or(Or((v["arg1_value"] == none), (v["arg2_value"] == 6)), (Or((And(v["arg1_value"] == True, v["arg2_value"] == 7)), (And(v["arg1_value"] == False, v["arg2_value"] == 8))))))
)

def rule_31_func(arg1, arg2, solver=None, neg=False):
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
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
