import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The length of the bytes tensor must be a multiple of the size of the out_type (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(Select(v["arg1_shape"], 0) % (If(v["arg2_value"] == 0, 1, If(Or(v["arg2_value"] == 1, v["arg2_value"] == 5), 1, If(v["arg2_value"] == 2, 2, If(v["arg2_value"] == 3, 4, If(v["arg2_value"] == 4, 8, If(v["arg2_value"] == 6, 2, If(v["arg2_value"] == 7, 4, If(v["arg2_value"] == 8, 8, If(v["arg2_value"] == 9, 8, If(v["arg2_value"] == 10, 16, 1))))))))))) == 0) if n else
          Select(v["arg1_shape"], 0) % (If(v["arg2_value"] == 0, 1, If(Or(v["arg2_value"] == 1, v["arg2_value"] == 5), 1, If(v["arg2_value"] == 2, 2, If(v["arg2_value"] == 3, 4, If(v["arg2_value"] == 4, 8, If(v["arg2_value"] == 6, 2, If(v["arg2_value"] == 7, 4, If(v["arg2_value"] == 8, 8, If(v["arg2_value"] == 9, 8, If(v["arg2_value"] == 10, 16, 1))))))))))) == 0)
)

def rule_60_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 60
        rule_60(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
