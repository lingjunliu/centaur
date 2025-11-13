import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If dense is not a scalar, it has elements in its dimensions, then dense and sp_shape must be broadcastable (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] != 0, (Or([And(k < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], k) > 0) for k in range(6)]))), And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Or(Or((v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i < 0), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == Select(v["arg2_shape"], i))), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == 1))) for i in range(6)]), True)) if n else
          If(And(v["arg1_ndim"] != 0, (Or([And(k < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], k) > 0) for k in range(6)]))), And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Or(Or((v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i < 0), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == Select(v["arg2_shape"], i))), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == 1))) for i in range(6)]), True))
)

def rule_74_func(arg1, arg2, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 74
        rule_74(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape']}, neg)
