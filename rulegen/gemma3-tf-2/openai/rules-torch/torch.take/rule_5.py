import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# indices valid for scalar input (Rule 5)

rule_5 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, And(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 1) == 0), True)) if n else
          If(v["arg1_ndim"] == 0, And(Select(v["arg2_range"], 0) == 0, Select(v["arg2_range"], 1) == 0), True))
)

def rule_5_func(arg1, arg2, solver=None, neg=False):
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
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 5
        rule_5(solver, {'arg1_ndim': arg1_ndim, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_5(solver, {'arg1_ndim': arg1['ndim'], 'arg2_range': arg2['range']}, neg)
