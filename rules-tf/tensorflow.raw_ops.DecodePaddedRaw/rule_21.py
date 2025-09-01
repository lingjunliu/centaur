import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# fixed_length must be a multiple of the out_type's size, which depends on the dtype (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(Or([And(size < (8 + 1), And((If(v["arg2_value"] == 6, size == 2, If(v["arg2_value"] == 7, size == 4, If(v["arg2_value"] == 8, size == 8, If(v["arg2_value"] == 3, size == 4, If(v["arg2_value"] == 5, size == 2, If(v["arg2_value"] == 2, size == 1, If(v["arg2_value"] == 1, size == 1, If(v["arg2_value"] == 4, size == 8, size == 4))))))))), Select(v["arg1_shape"], 0) % size == 0)) for size in range(6)])) if n else
          Or([And(size < (8 + 1), And((If(v["arg2_value"] == 6, size == 2, If(v["arg2_value"] == 7, size == 4, If(v["arg2_value"] == 8, size == 8, If(v["arg2_value"] == 3, size == 4, If(v["arg2_value"] == 5, size == 2, If(v["arg2_value"] == 2, size == 1, If(v["arg2_value"] == 1, size == 1, If(v["arg2_value"] == 4, size == 8, size == 4))))))))), Select(v["arg1_shape"], 0) % size == 0)) for size in range(6)]))
)

def rule_21_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 21
        rule_21(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value']}, neg)
