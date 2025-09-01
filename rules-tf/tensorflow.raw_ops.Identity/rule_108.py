import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tensor's dtype is string, its length should not be longer than a limit. (Rule 108)

rule_108 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], i) < 1000) for i in range(6)]), True)) if n else
          If(v["arg1_dtype"] == 11, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_shape"], i) < 1000) for i in range(6)]), True))
)

def rule_108_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 108
        rule_108(solver, {'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_108(solver, {'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
