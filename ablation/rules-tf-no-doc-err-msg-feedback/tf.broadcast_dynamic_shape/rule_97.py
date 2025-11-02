import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Number of float elements inside tuple(float (Rule 97)

rule_97 = lambda s, v, n=False: (
    s.add(Not(v["arg1_length"] < 1000) if n else
          v["arg1_length"] < 1000)
)

def rule_97_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))

        # Constraints for rule 97
        rule_97(solver, {'arg1_length': arg1_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_97(solver, {'arg1_length': arg1['length']}, neg)
