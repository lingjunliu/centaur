import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# no parameters permitted: any numeric (int or float (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] + 0 == v["arg1_value"], False, False)) if n else
          If(v["arg1_value"] + 0 == v["arg1_value"], False, False))
)

def rule_19_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value']}, neg)
