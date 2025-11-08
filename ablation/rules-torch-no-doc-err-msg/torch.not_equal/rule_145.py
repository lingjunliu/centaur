import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Must use supported value either reflect or replicate (Rule 145)

rule_145 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] == 22, v["arg1_value"] == 23)) if n else
          Or(v["arg1_value"] == 22, v["arg1_value"] == 23))
)

def rule_145_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 145
        rule_145(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_145(solver, {'arg1_value': arg1['value']}, neg)
