import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Dtype of input tensor and provided dtype should be same type if dtype is provided. For example, both should be integer or both should be float (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] != 12, (Or((And(v["arg1_dtype"] < 6, v["arg2_value"] < 6)), (And(v["arg1_dtype"] >= 6, v["arg2_value"] >= 6)))), False)) if n else
          If(v["arg2_value"] != 12, (Or((And(v["arg1_dtype"] < 6, v["arg2_value"] < 6)), (And(v["arg1_dtype"] >= 6, v["arg2_value"] >= 6)))), False))
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
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 21
        rule_21(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
