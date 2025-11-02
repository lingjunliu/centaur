import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# index tensor (v_2 (Rule 3)

rule_3 = lambda s, v, n=False: (
    s.add(Not(And(Select(v["arg2_range"], 0) >= 0, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], i) - 1) for i in range(6)]))) if n else
          And(Select(v["arg2_range"], 0) >= 0, And([Implies(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg2_range"], 1) <= Select(v["arg1_shape"], i) - 1) for i in range(6)])))
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
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 3
        rule_3(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_3(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range']}, neg)
