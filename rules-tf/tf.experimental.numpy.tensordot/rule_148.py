import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# axes must be a valid dimension on tensor b when it's a list (Rule 148)

rule_148 = lambda s, v, n=False: (
    s.add(Not(If((v["arg3_length"] == 2), (And(Select(v["arg3_values"], 1) >= 0, Select(v["arg3_values"], 1) < v["arg2_ndim"])), True)) if n else
          If((v["arg3_length"] == 2), (And(Select(v["arg3_values"], 1) >= 0, Select(v["arg3_values"], 1) < v["arg2_ndim"])), True))
)

def rule_148_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 148
        rule_148(solver, {'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length, 'arg3_values': arg3_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_148(solver, {'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length'], 'arg3_values': arg3['values']}, neg)
