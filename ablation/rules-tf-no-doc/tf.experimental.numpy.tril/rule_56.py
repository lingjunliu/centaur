import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the input tensor has different values in the rows, but k causes some to disappear, output is valid (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 1, True, True)) if n else
          If(v["arg1_ndim"] > 1, True, True))
)

def rule_56_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 56
        rule_56(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_ndim': arg1['ndim']}, neg)
