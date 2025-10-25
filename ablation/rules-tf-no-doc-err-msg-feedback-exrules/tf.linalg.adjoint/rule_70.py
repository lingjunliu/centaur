import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape must not be linear, softplus, swish, gelu, selu, elu, softmax, tanh. (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) != 20, Select(v["arg1_shape"], 0) != 19), Select(v["arg1_shape"], 0) != 18), Select(v["arg1_shape"], 0) != 17), Select(v["arg1_shape"], 0) != 16), Select(v["arg1_shape"], 0) != 15), Select(v["arg1_shape"], 0) != 14), Select(v["arg1_shape"], 0) != 12)) if n else
          And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) != 20, Select(v["arg1_shape"], 0) != 19), Select(v["arg1_shape"], 0) != 18), Select(v["arg1_shape"], 0) != 17), Select(v["arg1_shape"], 0) != 16), Select(v["arg1_shape"], 0) != 15), Select(v["arg1_shape"], 0) != 14), Select(v["arg1_shape"], 0) != 12))
)

def rule_70_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 70
        rule_70(solver, {'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_shape': arg1['shape']}, neg)
