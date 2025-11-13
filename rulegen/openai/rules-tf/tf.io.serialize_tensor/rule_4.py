import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dimension checks conditioned on having rank > 0 (Rule 4)

rule_4 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, And(Select(v["arg1_shape"], 0) >= 0, Select(v["arg1_shape"], v["arg1_ndim"] - 1) >= 0), True)) if n else
          If(v["arg1_ndim"] > 0, And(Select(v["arg1_shape"], 0) >= 0, Select(v["arg1_shape"], v["arg1_ndim"] - 1) >= 0), True))
)

def rule_4_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])

        # Constraints for rule 4
        rule_4(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_4(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim']}, neg)
