import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Shape compatibility between dense and sp_shape: dense must be able to be broadcasted to sp_shape. (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Or(Or((v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i < 0), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == Select(v["arg2_shape"], i))), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == 1))) for i in range(6)])) if n else
          And([Implies(i < (Select(v["arg2_shape"], 0) - 1 + 1), Or(Or((v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i < 0), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == Select(v["arg2_shape"], i))), (Select(v["arg1_shape"], v["arg1_ndim"] - Select(v["arg2_shape"], 0) + i) == 1))) for i in range(6)]))
)

def rule_32_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 32
        rule_32(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape']}, neg)
