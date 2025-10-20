import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shapes are broadcastable or the same shape or have an explicit batch dimension match (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(Or((Or((Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), (And([Implies(i < (If(v["arg1_ndim"] > v["arg2_ndim"], v["arg1_ndim"] - 1, v["arg2_ndim"] - 1) + 1), If(And(i < v["arg1_ndim"], i < v["arg2_ndim"]), (Or(Or(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == 1), Select(v["arg2_shape"], i) == 1)), True)) for i in range(6)])))), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)))) if n else
          Or((Or((Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0)), (And([Implies(i < (If(v["arg1_ndim"] > v["arg2_ndim"], v["arg1_ndim"] - 1, v["arg2_ndim"] - 1) + 1), If(And(i < v["arg1_ndim"], i < v["arg2_ndim"]), (Or(Or(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg1_shape"], i) == 1), Select(v["arg2_shape"], i) == 1)), True)) for i in range(6)])))), (Select(v["arg1_shape"], 0) == Select(v["arg2_shape"], 0))))
)

def rule_70_func(arg1, arg2, solver=None, neg=False):
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 70
        rule_70(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
