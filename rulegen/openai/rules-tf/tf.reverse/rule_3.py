import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# axis tensor values must be within valid range per input rank (Rule 3)

rule_3 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_ndim"] == 1, (If(Select(v["arg2_shape"], 0) == 0, True, (And(-1 * v["arg1_ndim"] <= Select(v["arg2_range"], 0), Select(v["arg2_range"], 1) <= v["arg1_ndim"] - 1)))))) if n else
          And(v["arg2_ndim"] == 1, (If(Select(v["arg2_shape"], 0) == 0, True, (And(-1 * v["arg1_ndim"] <= Select(v["arg2_range"], 0), Select(v["arg2_range"], 1) <= v["arg1_ndim"] - 1))))))
)

def rule_3_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 3
        rule_3(solver, {'arg1_ndim': arg1_ndim, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_3(solver, {'arg1_ndim': arg1['ndim'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
