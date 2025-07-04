import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# negative_slope should be a float32 or float64 (Rule 6)

rule_6 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0.0, v["arg1_value"] == 0.0001), v["arg1_value"] == 0.01), v["arg1_value"] == 0.1), v["arg1_value"] == 0.2), v["arg1_value"] == 0.3), v["arg1_value"] == 0.4), v["arg1_value"] == 0.5), v["arg1_value"] == 0.6), v["arg1_value"] == 0.7), v["arg1_value"] == 0.8), v["arg1_value"] == 0.9), v["arg1_value"] == 1.0)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 0.0, v["arg1_value"] == 0.0001), v["arg1_value"] == 0.01), v["arg1_value"] == 0.1), v["arg1_value"] == 0.2), v["arg1_value"] == 0.3), v["arg1_value"] == 0.4), v["arg1_value"] == 0.5), v["arg1_value"] == 0.6), v["arg1_value"] == 0.7), v["arg1_value"] == 0.8), v["arg1_value"] == 0.9), v["arg1_value"] == 1.0))
)

def rule_6_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 6
        rule_6(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_6(solver, {'arg1_value': arg1['value']}, neg)
