import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The saturation factor shouldn't be too close to zero, as it is likely that the user just wanted to convert to grayscale in that case. (Rule 75)

rule_75 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] < -0.2, v["arg1_value"] > 0.2)) if n else
          Or(v["arg1_value"] < -0.2, v["arg1_value"] > 0.2))
)

def rule_75_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 75
        rule_75(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_75(solver, {'arg1_value': arg1['value']}, neg)
