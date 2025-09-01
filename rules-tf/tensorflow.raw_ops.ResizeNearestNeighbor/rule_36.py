import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# images must have a valid dtype, batch and channels sizes must be greater than zero, height and width must be greater than 0 (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 14)), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 3) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0)) if n else
          And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 6), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 14)), Select(v["arg1_shape"], 0) > 0), Select(v["arg1_shape"], 3) > 0), Select(v["arg1_shape"], 1) > 0), Select(v["arg1_shape"], 2) > 0))
)

def rule_36_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 36
        rule_36(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
