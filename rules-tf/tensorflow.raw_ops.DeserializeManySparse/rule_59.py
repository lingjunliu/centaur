import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if serialized_sparse is non-empty and dtype is specified, then sparse_values dtype must match specified dtype (Rule 59)

rule_59 = lambda s, v, n=False: (
    s.add(Not(If(And(And(Select(v["arg1_shape"], 0) > 0, v["arg2_value"] >= 0), v["arg2_value"] <= 12), v["arg3_dtype"] == v["arg2_value"], False)) if n else
          If(And(And(Select(v["arg1_shape"], 0) > 0, v["arg2_value"] >= 0), v["arg2_value"] <= 12), v["arg3_dtype"] == v["arg2_value"], False))
)

def rule_59_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 59
        rule_59(solver, {'arg1_shape': arg1_shape, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_59(solver, {'arg1_shape': arg1['shape'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
