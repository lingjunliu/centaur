import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the mode is linear/bilinear/trilinear and recompute_scale_factor is true then align corners is required (Rule 133)

rule_133 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg1_value"] == 20, v["arg1_value"] == 26), v["arg1_value"] == 28)), v["arg2_value"] == True), (Or(v["arg3_value"] == True, v["arg3_value"] == False)), True)) if n else
          If(And((Or(Or(v["arg1_value"] == 20, v["arg1_value"] == 26), v["arg1_value"] == 28)), v["arg2_value"] == True), (Or(v["arg3_value"] == True, v["arg3_value"] == False)), True))
)

def rule_133_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 133
        rule_133(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_133(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
