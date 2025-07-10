import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# channels must be an integer, more grammar friendly (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (100000 + 1), v["arg1_value"] == i) for i in range(6)])) if n else
          Or([And(i < (100000 + 1), v["arg1_value"] == i) for i in range(6)]))
)

def rule_36_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value']}, neg)
