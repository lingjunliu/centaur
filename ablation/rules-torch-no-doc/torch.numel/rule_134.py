import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dimensions of a tensor is same as elements count of tuple and max value of tensor must be smaller than 1. (Rule 134)

rule_134 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == v["arg2_length"], Select(v["arg1_range"], 1) < 1)) if n else
          And(v["arg1_ndim"] == v["arg2_length"], Select(v["arg1_range"], 1) < 1))
)

def rule_134_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 134
        rule_134(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_134(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length']}, neg)
