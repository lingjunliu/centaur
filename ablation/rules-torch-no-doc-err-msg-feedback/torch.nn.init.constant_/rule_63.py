import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The length of the given tuple must be a prime number (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(v["arg1_length"] == 2, v["arg1_length"] == 3), v["arg1_length"] == 5), v["arg1_length"] == 7), v["arg1_length"] == 11), v["arg1_length"] == 13), v["arg1_length"] == 17), v["arg1_length"] == 19)) if n else
          Or(Or(Or(Or(Or(Or(Or(v["arg1_length"] == 2, v["arg1_length"] == 3), v["arg1_length"] == 5), v["arg1_length"] == 7), v["arg1_length"] == 11), v["arg1_length"] == 13), v["arg1_length"] == 17), v["arg1_length"] == 19))
)

def rule_63_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 63
        rule_63(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_length': arg1['length']}, neg)
