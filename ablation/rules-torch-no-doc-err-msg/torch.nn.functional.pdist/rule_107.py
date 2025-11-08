import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Float v_1 must be greater than 0 or smaller than -1 (Rule 107)

rule_107 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] > 0.0, v["arg1_value"] < -1.0)) if n else
          Or(v["arg1_value"] > 0.0, v["arg1_value"] < -1.0))
)

def rule_107_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 107
        rule_107(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_107(solver, {'arg1_value': arg1['value']}, neg)
