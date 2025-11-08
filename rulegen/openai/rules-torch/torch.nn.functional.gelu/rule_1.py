import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# approximate must be 'none' or 'tanh' (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] == 6, v["arg1_value"] == 12)) if n else
          Or(v["arg1_value"] == 6, v["arg1_value"] == 12))
)

def rule_1_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 1
        rule_1(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_value': arg1['value']}, neg)
