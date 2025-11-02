import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Input must be a tensor (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(False) if n else
          False)
)

def rule_63_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 63
        rule_63(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {}, neg)
