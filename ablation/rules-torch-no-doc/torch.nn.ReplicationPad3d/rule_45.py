import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input tensor should not have extremely large dimensions that lead to memory allocation errors (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(Select(v["arg1_shape"], 0) < 10000, Select(v["arg1_shape"], 1) < 10000), Select(v["arg1_shape"], 2) < 10000), Select(v["arg1_shape"], 3) < 10000), Select(v["arg1_shape"], 4) < 10000)) if n else
          And(And(And(And(Select(v["arg1_shape"], 0) < 10000, Select(v["arg1_shape"], 1) < 10000), Select(v["arg1_shape"], 2) < 10000), Select(v["arg1_shape"], 3) < 10000), Select(v["arg1_shape"], 4) < 10000))
)

def rule_45_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 45
        rule_45(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_shape': arg1['shape']}, neg)
