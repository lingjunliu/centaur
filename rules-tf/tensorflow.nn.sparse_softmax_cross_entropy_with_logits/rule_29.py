import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Validates ranks, shapes, and dtypes of labels and logits, handling scalar logits properly (Rule 29)

rule_29 = lambda s, v, n=False: (
    s.add(Not(And((And((Or((v["arg1_dtype"] == 4), (v["arg1_dtype"] == 5))), (Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8))))), (If((v["arg2_ndim"] == 0), (v["arg1_ndim"] == 0), (And((And((v["arg2_ndim"] >= 1), (v["arg1_ndim"] == v["arg2_ndim"] - 1))), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), (Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i))) for i in range(6)])))))))) if n else
          And((And((Or((v["arg1_dtype"] == 4), (v["arg1_dtype"] == 5))), (Or(Or((v["arg2_dtype"] == 6), (v["arg2_dtype"] == 7)), (v["arg2_dtype"] == 8))))), (If((v["arg2_ndim"] == 0), (v["arg1_ndim"] == 0), (And((And((v["arg2_ndim"] >= 1), (v["arg1_ndim"] == v["arg2_ndim"] - 1))), (And([Implies(i < (v["arg1_ndim"] - 1 + 1), (Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i))) for i in range(6)]))))))))
)

def rule_29_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 29
        rule_29(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_29(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
