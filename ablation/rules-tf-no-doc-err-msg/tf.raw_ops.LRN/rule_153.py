import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# check if shape of dimension 1 is greater than or equal to 4 and shape of dimension 2 is equal to 32 (Rule 153)

rule_153 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg1_shape"], 1) >= 4, Select(v["arg1_shape"], 2) == 32)) if n else
          And(Select(v["arg1_shape"], 1) >= 4, Select(v["arg1_shape"], 2) == 32))
)

def rule_153_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 153
        rule_153(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_153(solver, {'arg1_shape': arg1['shape']}, neg)
