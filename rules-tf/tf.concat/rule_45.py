import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# parameter 'axis' must be rank 0 (scalar), not rank 1 or higher.
# Error example: axis with shape (1,) → ValueError

rule_45 = lambda s, v, n=False: (
    s.add(Not(v["arg1_rank"] == 0)) if n else
    s.add(v["arg1_rank"] == 0)
)

def rule_45_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        solver = Solver()
        arg1_rank = Int('arg1_rank')

        solver.add(arg1_rank == arg1.ndim)

        rule_45(solver, {'arg1_rank': arg1_rank})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_rank': arg1['arg1_rank']}, neg)