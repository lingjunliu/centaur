import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# when you use stride parameter, this value must be integer (Rule 1134)

rule_1134 = lambda s, v, n=False: (
    s.add(Not((v["arg1_value"] / 1) * 1 == v["arg1_value"]) if n else
          (v["arg1_value"] / 1) * 1 == v["arg1_value"])
)

def rule_1134_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 1134
        rule_1134(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1134(solver, {'arg1_value': arg1['value']}, neg)
