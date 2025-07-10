import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If image height or width is too small, then channel size must be small (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(Or(Select(v["arg1_shape"], 1) < 5, Select(v["arg1_shape"], 2) < 5), Select(v["arg1_shape"], 3) < 10, False)) if n else
          If(Or(Select(v["arg1_shape"], 1) < 5, Select(v["arg1_shape"], 2) < 5), Select(v["arg1_shape"], 3) < 10, False))
)

def rule_22_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 22
        rule_22(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_shape': arg1['shape']}, neg)
