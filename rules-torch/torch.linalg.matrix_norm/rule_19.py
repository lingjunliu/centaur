import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if ord is int, it must be -2, -1, 1, or 2. (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(v["arg1_value"] == -2, v["arg1_value"] == -1), v["arg1_value"] == 1), v["arg1_value"] == 2)) if n else
          Or(Or(Or(v["arg1_value"] == -2, v["arg1_value"] == -1), v["arg1_value"] == 1), v["arg1_value"] == 2))
)

def rule_19_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value']}, neg)
