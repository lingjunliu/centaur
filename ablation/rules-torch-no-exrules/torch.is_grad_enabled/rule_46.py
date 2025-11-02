import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Output is boolean. (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(Or([And(x < (1 + 1), Or((And(x == 0, v["arg1_value"] == False)), (And(x == 1, v["arg1_value"] == True)))) for x in range(6)])) if n else
          Or([And(x < (1 + 1), Or((And(x == 0, v["arg1_value"] == False)), (And(x == 1, v["arg1_value"] == True)))) for x in range(6)]))
)

def rule_46_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 46
        rule_46(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_value': arg1['value']}, neg)
