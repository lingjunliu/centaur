import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a tensor has ndim=0, it is a scalar. Scalars are allowed, but the behavior is to multiply. The rule would just assert that if a tensor has ndim =0, then multiplying is allowed. (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, True, True)) if n else
          If(v["arg1_ndim"] == 0, True, True))
)

def rule_65_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 65
        rule_65(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_ndim': arg1['ndim']}, neg)
