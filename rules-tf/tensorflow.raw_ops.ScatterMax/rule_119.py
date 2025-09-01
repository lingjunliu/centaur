import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If ref has more than one dimension and indices has at least one dimension, then shape[update, ndim(indices (Rule 119)

rule_119 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] > 1, v["arg2_ndim"] > 0), And([Implies(i < (v["arg1_ndim"] - 1 - 1 + 1), Select(v["arg3_shape"], i + v["arg2_ndim"]) == Select(v["arg1_shape"], i + 1)) for i in range(6)]), True)) if n else
          If(And(v["arg1_ndim"] > 1, v["arg2_ndim"] > 0), And([Implies(i < (v["arg1_ndim"] - 1 - 1 + 1), Select(v["arg3_shape"], i + v["arg2_ndim"]) == Select(v["arg1_shape"], i + 1)) for i in range(6)]), True))
)

def rule_119_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 119
        rule_119(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg3_shape': arg3['shape']}, neg)
