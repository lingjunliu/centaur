import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The number of elements in the tensor shape cannot be zero (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(0 < Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) * Select(v["arg1_shape"], 4) * Select(v["arg1_shape"], 5) * Select(v["arg1_shape"], 6) * Select(v["arg1_shape"], 7)) if n else
          0 < Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) * Select(v["arg1_shape"], 2) * Select(v["arg1_shape"], 3) * Select(v["arg1_shape"], 4) * Select(v["arg1_shape"], 5) * Select(v["arg1_shape"], 6) * Select(v["arg1_shape"], 7))
)

def rule_36_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 36
        rule_36(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_shape': arg1['shape']}, neg)
