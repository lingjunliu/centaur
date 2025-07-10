import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The number of dimensions of the tensor is greater than or equal to 1 and less than or equal to 3. (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] >= 1, v["arg1_ndim"] <= 3)) if n else
          And(v["arg1_ndim"] >= 1, v["arg1_ndim"] <= 3))
)

def rule_29_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 29
        rule_29(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_ndim': arg1['ndim']}, neg)
