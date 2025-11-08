import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The string argument can only be from the available list, if the first variable is trilinear, bilinear, nearest, bicubic or area (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_value"] == 28, v["arg1_value"] == 26), v["arg1_value"] == 25), v["arg1_value"] == 27), v["arg1_value"] == 29), True, True)) if n else
          If(Or(Or(Or(Or(v["arg1_value"] == 28, v["arg1_value"] == 26), v["arg1_value"] == 25), v["arg1_value"] == 27), v["arg1_value"] == 29), True, True))
)

def rule_73_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))

        # Constraints for rule 73
        rule_73(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_value': arg1['value']}, neg)
