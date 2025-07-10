import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# x and y dimensions must be greater than 0 and less than 5 and x ndim less than y ndim and y shape 0 > 1 (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_ndim"] > 0, v["arg1_ndim"] < 5), v["arg2_ndim"] > 0), v["arg2_ndim"] < 5), v["arg1_ndim"] < v["arg2_ndim"]), Select(v["arg2_shape"], 0) > 1)) if n else
          And(And(And(And(And(v["arg1_ndim"] > 0, v["arg1_ndim"] < 5), v["arg2_ndim"] > 0), v["arg2_ndim"] < 5), v["arg1_ndim"] < v["arg2_ndim"]), Select(v["arg2_shape"], 0) > 1))
)

def rule_59_func(arg1, arg2, solver=None, neg=False):
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

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 59
        rule_59(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_shape': arg2['shape']}, neg)
