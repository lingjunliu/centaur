import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ksize's dimensions must be in range of orig_input dimensions (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), i < v["arg1_ndim"]) for i in range(6)])) if n else
          And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), i < v["arg1_ndim"]) for i in range(6)]))
)

def rule_102_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 102
        rule_102(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape']}, neg)
