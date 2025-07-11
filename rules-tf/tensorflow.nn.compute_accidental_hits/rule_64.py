import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# true_classes and sampled_candidates must have different shapes for the first dimension or the second dimension of true_classes should not be 1 and sample_candidates should not be empty (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(And((Or(Select(v["arg1_shape"], 0) != Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 1) != 1)), Select(v["arg2_shape"], 0) > 0)) if n else
          And((Or(Select(v["arg1_shape"], 0) != Select(v["arg2_shape"], 0), Select(v["arg1_shape"], 1) != 1)), Select(v["arg2_shape"], 0) > 0))
)

def rule_64_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 64
        rule_64(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape']}, neg)
