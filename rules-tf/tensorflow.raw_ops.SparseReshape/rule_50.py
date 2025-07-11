import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if any element of input indices has value -1, it must not be smaller than -1 (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(Or((Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), (Or([And(j < (Select(v["arg1_shape"], 1) - 1 + 1), Select(v["arg1_shape"], i) == -1) for j in range(6)]))) for i in range(6)])), Select(v["arg1_range"], 0) >= -1)) if n else
          Or((Or([And(i < (Select(v["arg1_shape"], 0) - 1 + 1), (Or([And(j < (Select(v["arg1_shape"], 1) - 1 + 1), Select(v["arg1_shape"], i) == -1) for j in range(6)]))) for i in range(6)])), Select(v["arg1_range"], 0) >= -1))
)

def rule_50_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 50
        rule_50(solver, {'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']}, neg)
