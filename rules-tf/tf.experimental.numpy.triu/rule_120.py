import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# parameter 'input' must have a statically known rank (i.e., must be an np.ndarray).
# Error example: triu with dynamic/unknown rank → ValueError

rule_120 = lambda s, v, n=False: (
    s.add(Not(v["arg1_rank"] >= 0)) if n else
    s.add(v["arg1_rank"] >= 0)
)

def rule_120_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        solver = Solver()
        arg1_rank = Int('arg1_rank')

        solver.add(arg1_rank == arg1.ndim)

        rule_120(solver, {'arg1_rank': arg1_rank})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_120(solver, {'arg1_rank': arg1['arg1_rank']}, neg)