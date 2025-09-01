import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Data tensor has supported dtype and a size constraint (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 6), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 11)), Select(v["arg1_shape"], 0) < 10000)) if n else
          And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 6), v["arg1_dtype"] == 14), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 17), v["arg1_dtype"] == 11)), Select(v["arg1_shape"], 0) < 10000))
)

def rule_54_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 54
        rule_54(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
