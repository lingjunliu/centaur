import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# max_attempts must be a positive integer and less than 1000 (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= 0, v["arg1_value"] < 1000)) if n else
          And(v["arg1_value"] >= 0, v["arg1_value"] < 1000))
)

def rule_66_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 66
        rule_66(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_value': arg1['value']}, neg)
