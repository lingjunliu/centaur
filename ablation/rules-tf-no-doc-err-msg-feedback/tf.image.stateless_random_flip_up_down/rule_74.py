import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Image size should be appropriate for the data type. (Rule 74)

rule_74 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 6, And(Select(v["arg1_shape"], 0) <= 65535, Select(v["arg1_shape"], 1) <= 65535), If(v["arg1_dtype"] == 5, And(Select(v["arg1_shape"], 0) <= 255, Select(v["arg1_shape"], 1) <= 255), True))) if n else
          If(v["arg1_dtype"] == 6, And(Select(v["arg1_shape"], 0) <= 65535, Select(v["arg1_shape"], 1) <= 65535), If(v["arg1_dtype"] == 5, And(Select(v["arg1_shape"], 0) <= 255, Select(v["arg1_shape"], 1) <= 255), True)))
)

def rule_74_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 74
        rule_74(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_74(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype']}, neg)
