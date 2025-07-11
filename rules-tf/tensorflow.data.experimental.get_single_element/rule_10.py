import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the shape of single element tensor is 0, then dtype should not be int (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) == 0, And(And(And(And(v["arg1_dtype"] != 1, v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), False)) if n else
          If(Select(v["arg1_shape"], 0) == 0, And(And(And(And(v["arg1_dtype"] != 1, v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), False))
)

def rule_10_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 10
        rule_10(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
