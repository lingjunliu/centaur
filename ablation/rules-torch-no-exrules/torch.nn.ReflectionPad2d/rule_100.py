import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The number of dimensions of the input tensor must be between 2 and 5 inclusive to be supported by the functional implementation (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_ndim"] >= 2), (v["arg1_ndim"] <= 5))) if n else
          And((v["arg1_ndim"] >= 2), (v["arg1_ndim"] <= 5)))
)

def rule_100_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 100
        rule_100(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_ndim': arg1['ndim']}, neg)
