import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if both size_average and reduce are specified, their values should correspond to the value of reduction (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_value"] == none, v["arg2_value"] == none)), (Or(Or((And(And(v["arg1_value"] == True, v["arg2_value"] == True), v["arg3_value"] == 7)), (And(And(v["arg1_value"] == False, v["arg2_value"] == False), v["arg3_value"] == 8))), (And(And(v["arg1_value"] == none, v["arg2_value"] == none), v["arg3_value"] == 6)))))) if n else
          Or((And(v["arg1_value"] == none, v["arg2_value"] == none)), (Or(Or((And(And(v["arg1_value"] == True, v["arg2_value"] == True), v["arg3_value"] == 7)), (And(And(v["arg1_value"] == False, v["arg2_value"] == False), v["arg3_value"] == 8))), (And(And(v["arg1_value"] == none, v["arg2_value"] == none), v["arg3_value"] == 6))))))
)

def rule_42_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or isinstance(arg1, str)):
            return False
        if not (isinstance(arg2, bool) or isinstance(arg2, str)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
