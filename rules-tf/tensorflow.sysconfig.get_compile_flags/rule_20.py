import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The length of the list is either 2 or 4 or 6 (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_length"] == 2, v["arg1_length"] == 4), v["arg1_length"] == 6)) if n else
          Or(Or(v["arg1_length"] == 2, v["arg1_length"] == 4), v["arg1_length"] == 6))
)

def rule_20_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 20
        rule_20(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_length': arg1['length']}, neg)
