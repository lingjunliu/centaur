import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The API can return one boolean value, i.e., the variable should have only one element (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (0 + 1), Or(v["arg1_value"] == True, v["arg1_value"] == False)) for i in range(6)])) if n else
          Or([And(i < (0 + 1), Or(v["arg1_value"] == True, v["arg1_value"] == False)) for i in range(6)]))
)

def rule_13_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 13
        rule_13(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_value': arg1['value']}, neg)
