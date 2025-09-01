import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input must be a tensor, such as torch.Tensor. (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (1 + 1), v["arg1_ndim"] >= 0) for i in range(6)])) if n else
          Or([And(i < (1 + 1), v["arg1_ndim"] >= 0) for i in range(6)]))
)

def rule_66_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 66
        rule_66(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_ndim': arg1['ndim']}, neg)
