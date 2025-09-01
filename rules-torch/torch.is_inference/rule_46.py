import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dimension is 0, shape should not be called (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, True, Or([And(i < (v["arg1_ndim"] - 1 + 1), True) for i in range(6)]))) if n else
          If(v["arg1_ndim"] == 0, True, Or([And(i < (v["arg1_ndim"] - 1 + 1), True) for i in range(6)])))
)

def rule_46_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 46
        rule_46(solver, {'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_ndim': arg1['ndim']}, neg)
