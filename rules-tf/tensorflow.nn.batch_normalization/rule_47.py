import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x's shape is [0,37,31,91,38] variance's shape must be [14,56,57] or broadcastable to x (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(And(And(And(And(Select(v["arg1_shape"], 0) == 0, Select(v["arg1_shape"], 1) == 37), Select(v["arg1_shape"], 2) == 31), Select(v["arg1_shape"], 3) == 91), Select(v["arg1_shape"], 4) == 38), Or((And(And(Select(v["arg2_shape"], 0) == 14, Select(v["arg2_shape"], 1) == 56), Select(v["arg2_shape"], 2) == 57)), (And(And(v["arg2_ndim"] > 0, v["arg1_ndim"] > 0), And([Implies(i < (If(v["arg2_ndim"] >= v["arg1_ndim"], v["arg2_ndim"] - 1, v["arg1_ndim"] - 1) + 1), Or(Or(Or(Or(v["arg2_ndim"] - i - 1 < 0, v["arg1_ndim"] - i - 1 < 0), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1), Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == Select(v["arg1_shape"], v["arg1_ndim"] - i - 1))) for i in range(6)])))), True)) if n else
          If(And(And(And(And(Select(v["arg1_shape"], 0) == 0, Select(v["arg1_shape"], 1) == 37), Select(v["arg1_shape"], 2) == 31), Select(v["arg1_shape"], 3) == 91), Select(v["arg1_shape"], 4) == 38), Or((And(And(Select(v["arg2_shape"], 0) == 14, Select(v["arg2_shape"], 1) == 56), Select(v["arg2_shape"], 2) == 57)), (And(And(v["arg2_ndim"] > 0, v["arg1_ndim"] > 0), And([Implies(i < (If(v["arg2_ndim"] >= v["arg1_ndim"], v["arg2_ndim"] - 1, v["arg1_ndim"] - 1) + 1), Or(Or(Or(Or(v["arg2_ndim"] - i - 1 < 0, v["arg1_ndim"] - i - 1 < 0), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1), Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1), Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == Select(v["arg1_shape"], v["arg1_ndim"] - i - 1))) for i in range(6)])))), True))
)

def rule_47_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 47
        rule_47(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
