import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If size_average is None or not specified, reduce should be 'mean', 'sum' or 'none' (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == False), v["arg1_value"] == True), Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 6), True)) if n else
          If(Or(Or(v["arg1_value"] == 6, v["arg1_value"] == False), v["arg1_value"] == True), Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 6), True))
)

def rule_30_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
