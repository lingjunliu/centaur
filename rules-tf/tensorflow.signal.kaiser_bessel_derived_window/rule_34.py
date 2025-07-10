import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# dtype is either float16, float32, or float64, but it is also not a boolean (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8)), v["arg1_value"] != 0)) if n else
          And((Or(Or(v["arg1_value"] == 6, v["arg1_value"] == 7), v["arg1_value"] == 8)), v["arg1_value"] != 0))
)

def rule_34_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 34
        rule_34(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_value': arg1['value']}, neg)
