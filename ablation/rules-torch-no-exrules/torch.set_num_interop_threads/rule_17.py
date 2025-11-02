import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# num_threads must be an integer or can be converted to an integer (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == v["arg1_value"], True, False)) if n else
          If(v["arg1_value"] == v["arg1_value"], True, False))
)

def rule_17_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating)) or isinstance(arg1, str) or isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 17
        rule_17(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_value': arg1['value']}, neg)
