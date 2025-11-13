import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# axes must be distinct even after negative-index normalization (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) != Select(v["arg2_values"], 1)), Select(v["arg2_values"], 0) + v["arg1_ndim"] != Select(v["arg2_values"], 1)), Select(v["arg2_values"], 1) + v["arg1_ndim"] != Select(v["arg2_values"], 0))) if n else
          And(And(And(v["arg2_length"] == 2, Select(v["arg2_values"], 0) != Select(v["arg2_values"], 1)), Select(v["arg2_values"], 0) + v["arg1_ndim"] != Select(v["arg2_values"], 1)), Select(v["arg2_values"], 1) + v["arg1_ndim"] != Select(v["arg2_values"], 0)))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
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
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 15
        rule_15(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
