import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# output_size tuple must contain exactly 3 elements (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(v["arg1_length"] == 3) if n else
          v["arg1_length"] == 3)
)

def rule_2_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 2
        rule_2(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_length': arg1['length']}, neg)
