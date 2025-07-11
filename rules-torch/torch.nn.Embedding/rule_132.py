import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Norm_type needs a limit of a smaller range - new number  (Rule 132)

rule_132 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] < 200) if n else
          v["arg1_value"] < 200)
)

def rule_132_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 132
        rule_132(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_132(solver, {'arg1_value': arg1['value']}, neg)
