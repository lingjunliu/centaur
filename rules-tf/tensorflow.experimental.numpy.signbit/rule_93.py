import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first dimension exists then it must be different than tanh, sigmoid, softmax and other strings (Rule 93)

rule_93 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, And(And(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) != 12, Select(v["arg1_shape"], 0) != 13), Select(v["arg1_shape"], 0) != 14), Select(v["arg1_shape"], 0) != 15), Select(v["arg1_shape"], 0) != 16), Select(v["arg1_shape"], 0) != 17), Select(v["arg1_shape"], 0) != 18), Select(v["arg1_shape"], 0) != 19), Select(v["arg1_shape"], 0) != 20), Select(v["arg1_shape"], 0) != 21), Select(v["arg1_shape"], 0) != 22), Select(v["arg1_shape"], 0) != 23), Select(v["arg1_shape"], 0) != 24), Select(v["arg1_shape"], 0) != 25), True)) if n else
          If(v["arg1_ndim"] > 0, And(And(And(And(And(And(And(And(And(And(And(And(And(Select(v["arg1_shape"], 0) != 12, Select(v["arg1_shape"], 0) != 13), Select(v["arg1_shape"], 0) != 14), Select(v["arg1_shape"], 0) != 15), Select(v["arg1_shape"], 0) != 16), Select(v["arg1_shape"], 0) != 17), Select(v["arg1_shape"], 0) != 18), Select(v["arg1_shape"], 0) != 19), Select(v["arg1_shape"], 0) != 20), Select(v["arg1_shape"], 0) != 21), Select(v["arg1_shape"], 0) != 22), Select(v["arg1_shape"], 0) != 23), Select(v["arg1_shape"], 0) != 24), Select(v["arg1_shape"], 0) != 25), True))
)

def rule_93_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 93
        rule_93(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_93(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
