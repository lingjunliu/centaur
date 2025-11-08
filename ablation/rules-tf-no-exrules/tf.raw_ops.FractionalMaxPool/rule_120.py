import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the length of pooling_ratio is not 4, 5, 6 and greater than 3, then return false (Rule 120)

rule_120 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] > 3, Or(Or(v["arg1_length"] == 4, v["arg1_length"] == 5), v["arg1_length"] == 6), True)) if n else
          If(v["arg1_length"] > 3, Or(Or(v["arg1_length"] == 4, v["arg1_length"] == 5), v["arg1_length"] == 6), True))
)

def rule_120_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 120
        rule_120(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_120(solver, {'arg1_length': arg1['length']}, neg)
