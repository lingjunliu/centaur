import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Ensure out tensor's type matches expectation based on input and dtype (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == None, (If(v["arg1_dtype"] == 6, v["arg2_dtype"] == 6, If(v["arg1_dtype"] == 7, v["arg2_dtype"] == 7, If(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8, If(v["arg1_dtype"] == 9, v["arg2_dtype"] == 9, If(v["arg1_dtype"] == 10, v["arg2_dtype"] == 10, True)))))), v["arg2_dtype"] == v["arg3_value"])) if n else
          If(v["arg3_value"] == None, (If(v["arg1_dtype"] == 6, v["arg2_dtype"] == 6, If(v["arg1_dtype"] == 7, v["arg2_dtype"] == 7, If(v["arg1_dtype"] == 8, v["arg2_dtype"] == 8, If(v["arg1_dtype"] == 9, v["arg2_dtype"] == 9, If(v["arg1_dtype"] == 10, v["arg2_dtype"] == 10, True)))))), v["arg2_dtype"] == v["arg3_value"]))
)

def rule_60_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, torch.dtype) or isinstance(arg3, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == list_of_available_dtypes.index(np_dtype(arg3)))

        # Constraints for rule 60
        rule_60(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
