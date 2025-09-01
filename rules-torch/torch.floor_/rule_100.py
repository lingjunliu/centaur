import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Checking data type as int or float , and having values in dimensions. (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, And((And([Implies(j < (If(v["arg1_ndim"] - 1 < 0, 0, v["arg1_ndim"] - 1) + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)])), (And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 9)))), True)) if n else
          If(v["arg1_ndim"] > 0, And((And([Implies(j < (If(v["arg1_ndim"] - 1 < 0, 0, v["arg1_ndim"] - 1) + 1), Select(v["arg1_shape"], j) > 0) for j in range(6)])), (And((v["arg1_dtype"] > 0), (v["arg1_dtype"] < 9)))), True))
)

def rule_100_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 100
        rule_100(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
