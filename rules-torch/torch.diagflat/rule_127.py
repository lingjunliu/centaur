import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# More stringent limits on dimension when its beyond 3 (Rule 127)

rule_127 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 3, (And(And(And(Select(v["arg1_shape"], 0) < 2000, Select(v["arg1_shape"], 1) < 2000), Select(v["arg1_shape"], 2) < 2000), Select(v["arg1_shape"], 3) < 2000)), True)) if n else
          If(v["arg1_ndim"] > 3, (And(And(And(Select(v["arg1_shape"], 0) < 2000, Select(v["arg1_shape"], 1) < 2000), Select(v["arg1_shape"], 2) < 2000), Select(v["arg1_shape"], 3) < 2000)), True))
)

def rule_127_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 127
        rule_127(solver, {'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_127(solver, {'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
