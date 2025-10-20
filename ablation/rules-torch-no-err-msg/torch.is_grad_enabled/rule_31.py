import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# is_grad_enabled returns a boolean, and any arithmetic on bools will not be performed (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, 1 + 1, Or(0 + 0 == 2, If(v["arg1_value"] == True, 1 + 1, 0 + 0 == 0)))) if n else
          If(v["arg1_value"] == True, 1 + 1, Or(0 + 0 == 2, If(v["arg1_value"] == True, 1 + 1, 0 + 0 == 0))))
)

def rule_31_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 31
        rule_31(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_value': arg1['value']}, neg)
