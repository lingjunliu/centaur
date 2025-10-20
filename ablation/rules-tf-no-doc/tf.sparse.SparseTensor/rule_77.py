import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape dimension at index 1 equal 68, then shape dimensions at index 2 must be greater than 70. (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 1) == 68, Select(v["arg1_shape"], 2) > 70, True)) if n else
          If(Select(v["arg1_shape"], 1) == 68, Select(v["arg1_shape"], 2) > 70, True))
)

def rule_77_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 77
        rule_77(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_shape': arg1['shape']}, neg)
