import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check to prevent big types being passed the values should be contained. If dimension exists and there does exist one of those numbers then maximum is smaller than 2 (Rule 126)

rule_126 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, If(Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 2) for i in range(6)]), Select(v["arg1_range"], 1) < 2, True), True)) if n else
          If(v["arg1_ndim"] > 0, If(Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) < 2) for i in range(6)]), Select(v["arg1_range"], 1) < 2, True), True))
)

def rule_126_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 126
        rule_126(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_126(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
