import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# LU_data's second to last dimension must be positive and it should be equal to the length of LU_pivots pivots are between 1 and LU_data.size(-2 (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg1_ndim"] - 2)), Select(v["arg2_range"], 0) >= 1), Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], v["arg1_ndim"] - 2))) if n else
          And(And(And(Select(v["arg1_shape"], v["arg1_ndim"] - 2) > 0, Select(v["arg2_shape"], 0) == Select(v["arg1_shape"], v["arg1_ndim"] - 2)), Select(v["arg2_range"], 0) >= 1), Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], v["arg1_ndim"] - 2)))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 63
        rule_63(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range']}, neg)
