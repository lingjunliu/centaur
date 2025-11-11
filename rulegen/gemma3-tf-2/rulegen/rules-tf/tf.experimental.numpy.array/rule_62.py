import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# ndim is between 1 and 5 inclusive (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(And(1 <= v["arg1_ndim"], v["arg1_ndim"] <= 5)) if n else
          And(1 <= v["arg1_ndim"], v["arg1_ndim"] <= 5))
)

def rule_62_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 62
        rule_62(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_ndim': arg1['ndim']}, neg)
