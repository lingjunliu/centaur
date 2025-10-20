import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if all shape values are equivalent then shape values should be valid or equal to 1. (Rule 112)

rule_112 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], i), Select(v["arg1_shape"], i) >= 1, True)) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_ndim"] - 1 + 1), If(Select(v["arg1_shape"], 0) == Select(v["arg1_shape"], i), Select(v["arg1_shape"], i) >= 1, True)) for i in range(6)]))
)

def rule_112_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 112
        rule_112(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_112(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
