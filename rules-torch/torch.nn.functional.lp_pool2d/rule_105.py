import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input is empty then the result is empty (Rule 105)

rule_105 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Select(v["arg1_shape"], 0) == 0, Select(v["arg1_shape"], 1) == 0), Select(v["arg1_shape"], 2) == 0), Select(v["arg1_shape"], 3) == 0)) if n else
          Or(Or(Or(Select(v["arg1_shape"], 0) == 0, Select(v["arg1_shape"], 1) == 0), Select(v["arg1_shape"], 2) == 0), Select(v["arg1_shape"], 3) == 0))
)

def rule_105_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 105
        rule_105(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_105(solver, {'arg1_shape': arg1['shape']}, neg)
