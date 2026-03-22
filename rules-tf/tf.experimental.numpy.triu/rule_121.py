import numpy as np
import torch
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The last two dimensions of the input must be statically known (concrete, > 0).
# Error example: triu with unknown last two dimensions → ValueError

rule_121 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_last_dim"] > 0, v["arg1_second_last_dim"] > 0))) if n else
    s.add(And(v["arg1_last_dim"] > 0, v["arg1_second_last_dim"] > 0))
)

def rule_121_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if arg1.ndim < 2:
            return False

        solver = Solver()
        arg1_last_dim = Int('arg1_last_dim')
        arg1_second_last_dim = Int('arg1_second_last_dim')

        solver.add(arg1_last_dim == arg1.shape[-1])
        solver.add(arg1_second_last_dim == arg1.shape[-2])

        rule_121(solver, {
            'arg1_last_dim': arg1_last_dim,
            'arg1_second_last_dim': arg1_second_last_dim
        })
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {
            'arg1_last_dim': arg1['arg1_last_dim'],
            'arg1_second_last_dim': arg1['arg1_second_last_dim']
        }, neg)