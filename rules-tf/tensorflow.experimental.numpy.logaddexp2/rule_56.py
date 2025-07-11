import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first dimension of first tensor is greater than 100, second tensor cannot be an integer tensor (Rule 56)

rule_56 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > 100, And(And(And(And((v["arg2_dtype"] != 1), (v["arg2_dtype"] != 2)), (v["arg2_dtype"] != 3)), (v["arg2_dtype"] != 4)), (v["arg2_dtype"] != 5)), False)) if n else
          If(Select(v["arg1_shape"], 0) > 100, And(And(And(And((v["arg2_dtype"] != 1), (v["arg2_dtype"] != 2)), (v["arg2_dtype"] != 3)), (v["arg2_dtype"] != 4)), (v["arg2_dtype"] != 5)), False))
)

def rule_56_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 56
        rule_56(solver, {'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_56(solver, {'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype']}, neg)
