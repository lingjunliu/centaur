import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check if handle_thread_failure function's argument is an exception (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(True) if n else
          True)
)

def rule_31_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 31
        rule_31(solver, {})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {}, neg)
