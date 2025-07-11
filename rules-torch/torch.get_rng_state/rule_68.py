import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The shape of tensor can only be 1024, 2048, 4096 (Rule 68)

rule_68 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Select(v["arg1_shape"], 0) == 1024, Select(v["arg1_shape"], 0) == 2048), Select(v["arg1_shape"], 0) == 4096)) if n else
          Or(Or(Select(v["arg1_shape"], 0) == 1024, Select(v["arg1_shape"], 0) == 2048), Select(v["arg1_shape"], 0) == 4096))
)

def rule_68_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 68
        rule_68(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_68(solver, {'arg1_shape': arg1['shape']}, neg)
