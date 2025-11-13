import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# s must not be np.nan (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(False) if n else
          False)
)

def rule_108_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 108
        rule_108(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {}, neg)
