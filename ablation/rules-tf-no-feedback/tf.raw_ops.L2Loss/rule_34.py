import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Tensor t's shape must be representable as a tuple of int64 (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(True) if n else
          True)
)

def rule_34_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 34
        rule_34(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {}, neg)
