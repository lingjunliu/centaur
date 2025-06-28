import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Each element of the repeats tensor v_2 must be non-negative (Rule 1249)

rule_1249 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg1_shape"], i) >= 0) for i in range(6)])) if n else
          And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg1_shape"], i) >= 0) for i in range(6)]))
)

def rule_1249_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 1249
        rule_1249(solver, {'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1249(solver, {'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim']}, neg)
