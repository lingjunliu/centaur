import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# prevent momentum from being able to completely overshadow old value from running variance (Rule 156)

rule_156 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] < 0.99999999) if n else
          v["arg1_value"] < 0.99999999)
)

def rule_156_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 156
        rule_156(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_156(solver, {'arg1_value': arg1['value']}, neg)
