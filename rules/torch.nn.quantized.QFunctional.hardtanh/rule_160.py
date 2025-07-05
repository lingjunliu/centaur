import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Shape of the tensor has to be greater than 1 in at least one dimension and the other lower or equal to 1.  (Rule 160)

rule_160 = lambda s, v, n=False: (
    s.add(Not(Or((And(Select(v["arg1_shape"], 0) > 1, Select(v["arg1_shape"], 1) <= 1)), (And(Select(v["arg1_shape"], 1) > 1, Select(v["arg1_shape"], 0) <= 1)))) if n else
          Or((And(Select(v["arg1_shape"], 0) > 1, Select(v["arg1_shape"], 1) <= 1)), (And(Select(v["arg1_shape"], 1) > 1, Select(v["arg1_shape"], 0) <= 1))))
)

def rule_160_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 160
        rule_160(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_160(solver, {'arg1_shape': arg1['shape']}, neg)
