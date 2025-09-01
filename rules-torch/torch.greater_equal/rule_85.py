import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Sizes should be broadcast compatible and must have consistent dtype (or both are numeric (Rule 85)

rule_85 = lambda s, v, n=False: (
    s.add(Not(And((And((And((v["arg1_dtype"] < 9), (v["arg2_dtype"] < 9))), (Or(Or((v["arg1_dtype"] == v["arg2_dtype"]), (And((v["arg1_dtype"] < 6), (v["arg2_dtype"] < 6)))), (And(And(And((v["arg1_dtype"] > 5), (v["arg1_dtype"] < 9)), (v["arg2_dtype"] > 5)), (v["arg2_dtype"] < 9))))))), (And([Implies(i < (If(v["arg1_ndim"] > v["arg2_ndim"], v["arg1_ndim"], v["arg2_ndim"] - 1) + 1), (If(And((v["arg1_ndim"] - i - 1) >= 0, (v["arg2_ndim"] - i - 1) >= 0), Or(Or((Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - i - 1)), (Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1)), (Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1)), True))) for i in range(6)])))) if n else
          And((And((And((v["arg1_dtype"] < 9), (v["arg2_dtype"] < 9))), (Or(Or((v["arg1_dtype"] == v["arg2_dtype"]), (And((v["arg1_dtype"] < 6), (v["arg2_dtype"] < 6)))), (And(And(And((v["arg1_dtype"] > 5), (v["arg1_dtype"] < 9)), (v["arg2_dtype"] > 5)), (v["arg2_dtype"] < 9))))))), (And([Implies(i < (If(v["arg1_ndim"] > v["arg2_ndim"], v["arg1_ndim"], v["arg2_ndim"] - 1) + 1), (If(And((v["arg1_ndim"] - i - 1) >= 0, (v["arg2_ndim"] - i - 1) >= 0), Or(Or((Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == Select(v["arg2_shape"], v["arg2_ndim"] - i - 1)), (Select(v["arg1_shape"], v["arg1_ndim"] - i - 1) == 1)), (Select(v["arg2_shape"], v["arg2_ndim"] - i - 1) == 1)), True))) for i in range(6)]))))
)

def rule_85_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 85
        rule_85(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_85(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
