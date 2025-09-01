import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# axis must be 0 or -1, or None which defaults to 0 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(Or(Or((v["arg1_value"] == 0), (v["arg1_value"] == -1)), (v["arg1_value"] == 6))) if n else
          Or(Or((v["arg1_value"] == 0), (v["arg1_value"] == -1)), (v["arg1_value"] == 6)))
)

def rule_20_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value']}, neg)
