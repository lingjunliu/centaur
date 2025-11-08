import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# tensor dimensions must be reasonable - version 4, also dimension 1 need to be bigger than 1 and less than 10 and at least one dimension is bigger than 5 (Rule 132)

rule_132 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(0 < v["arg1_ndim"], v["arg1_ndim"] < 5), Select(v["arg1_shape"], 0) > 1), Select(v["arg1_shape"], 0) < 10), Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 5) for i in range(6)]))) if n else
          And(And(And(And(0 < v["arg1_ndim"], v["arg1_ndim"] < 5), Select(v["arg1_shape"], 0) > 1), Select(v["arg1_shape"], 0) < 10), Or([And(i < (v["arg1_ndim"] - 1 + 1), Select(v["arg1_shape"], i) > 5) for i in range(6)])))
)

def rule_132_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 132
        rule_132(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_132(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
