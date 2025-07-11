import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape of v1 is greather then 5 then shape of v2 has to be smaller then 5 (Rule 102)

rule_102 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_shape"], 0) > 5, Select(v["arg2_shape"], 0) < 5, False)) if n else
          If(Select(v["arg1_shape"], 0) > 5, Select(v["arg2_shape"], 0) < 5, False))
)

def rule_102_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 102
        rule_102(solver, {'arg1_shape': arg1_shape, 'arg2_shape': arg2_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_102(solver, {'arg1_shape': arg1['shape'], 'arg2_shape': arg2['shape']}, neg)
