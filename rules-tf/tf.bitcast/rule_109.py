import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tf.bitcast rank rule:
# Input tensor must be at least rank 1 (no scalars allowed).
# Error example: bitcast from int16 to float32 with input shape [] (rank 0) → ValueError

rule_109 = lambda s, v, n=False: (
    s.add(Not(v["arg1_rank"] >= 1)) if n else
    s.add(v["arg1_rank"] >= 1)
)

def rule_109_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        rank = arg1.ndim

        solver = Solver()
        arg1_rank = Int('arg1_rank')

        solver.add(arg1_rank == rank)

        rule_109(solver, {'arg1_rank': arg1_rank})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_rank': arg1['arg1_rank']}, neg)