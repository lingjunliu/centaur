import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Tensor should be greater than value in list if tensor dimensions are greater than two (Rule 119)

rule_119 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 2, Select(v["arg2_values"], 0) < Select(v["arg1_range"], 1), True)) if n else
          If(v["arg1_ndim"] > 2, Select(v["arg2_values"], 0) < Select(v["arg1_range"], 1), True))
)

def rule_119_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 119
        rule_119(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_values': arg2['values']}, neg)
