import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If one tensor is empty, the other should also be empty or have shape that allows broadcasting (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, Or(v["arg2_ndim"] == 0, (And(v["arg2_ndim"] > 0, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]))))), False)) if n else
          If(v["arg1_ndim"] == 0, Or(v["arg2_ndim"] == 0, (And(v["arg2_ndim"] > 0, (And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_shape"], i) == 1) for i in range(6)]))))), False))
)

def rule_11_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 11
        rule_11(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
