import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# is_grad_enabled returns a boolean, so it's either true or false, and false can be expressed as not true (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] == True, v["arg1_value"] == (If(True == True, False, True)))) if n else
          Or(v["arg1_value"] == True, v["arg1_value"] == (If(True == True, False, True))))
)

def rule_30_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 30
        rule_30(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_value': arg1['value']}, neg)
