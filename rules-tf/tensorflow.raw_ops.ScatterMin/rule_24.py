import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If indices is a matrix and updates is not a scalar, then shape of updates from 2 to end must match shape of ref from 1 to end (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_ndim"] == 2, v["arg2_ndim"] > 0), And((v["arg2_ndim"] - 2) == (v["arg1_ndim"] - 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg2_shape"], i + 1) == Select(v["arg1_shape"], i)) for i in range(6)]))), False)) if n else
          If(And(v["arg3_ndim"] == 2, v["arg2_ndim"] > 0), And((v["arg2_ndim"] - 2) == (v["arg1_ndim"] - 1), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg2_shape"], i + 1) == Select(v["arg1_shape"], i)) for i in range(6)]))), False))
)

def rule_24_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 24
        rule_24(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim']}, neg)
