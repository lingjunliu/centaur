import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if list size is one and dimension of the tensor is less than 3, max of the tensor must be larger than 0 (Rule 96)

rule_96 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_length"] == 1, v["arg1_ndim"] < 3), Select(v["arg1_range"], 1) > 0, True)) if n else
          If(And(v["arg2_length"] == 1, v["arg1_ndim"] < 3), Select(v["arg1_range"], 1) > 0, True))
)

def rule_96_func(arg1, arg2, solver=None, neg=False):
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
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 96
        rule_96(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_96(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_length': arg2['length']}, neg)
