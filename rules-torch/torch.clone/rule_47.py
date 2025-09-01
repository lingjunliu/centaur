import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If tensor's datatype is str, the string length cannot be too long (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_range"], 1) < 100) for i in range(6)]), True)) if n else
          If(v["arg1_dtype"] == 11, And([Implies(i < (Select(v["arg1_shape"], 0) - 1 + 1), Select(v["arg1_range"], 1) < 100) for i in range(6)]), True))
)

def rule_47_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 47
        rule_47(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape']}, neg)
