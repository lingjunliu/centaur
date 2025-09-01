import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# input tensor must be broadcastable with b x n x p, given batch1 is b x n x m and batch2 is b x m x p. Covers all broadcasting possibilities and all required dimensions (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), (Or(Or(Or(Or(v["arg1_ndim"] - i - 1 < 0, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1), (And(i == 0, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], 0)))), (And(i == 1, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], 1)))), (And(i == 2, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg3_shape"], 2)))))) for i in range(6)])) if n else
          And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), (Or(Or(Or(Or(v["arg1_ndim"] - i - 1 < 0, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1), (And(i == 0, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], 0)))), (And(i == 1, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], 1)))), (And(i == 2, Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg3_shape"], 2)))))) for i in range(6)]))
)

def rule_74_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg3_shape = Array('arg3_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        for i in range(arg3.ndim):
            arg3_shape = Store(arg3_shape, i, arg3.shape[i])

        # Constraints for rule 74
        rule_74(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg3_shape': arg3_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg3_shape': arg3['shape']}, neg)
