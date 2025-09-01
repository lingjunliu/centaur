import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If probability is not a float between 0 and 1 throw value error (Rule 35)

rule_35 = lambda s, v, n=False: (
    s.add(Not(If(Or((v["arg1_value"] < 0.0), (v["arg1_value"] > 1.0)), Or([And(x < (0 + 1), False) for x in range(6)]), True)) if n else
          If(Or((v["arg1_value"] < 0.0), (v["arg1_value"] > 1.0)), Or([And(x < (0 + 1), False) for x in range(6)]), True))
)

def rule_35_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 35
        rule_35(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_35(solver, {'arg1_value': arg1['value']}, neg)
