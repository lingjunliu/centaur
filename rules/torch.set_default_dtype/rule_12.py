import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# The type promotion between bool/int and Python float/complex number depends on default dtype v_1 (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 6, If(v["arg2_dtype"] == 6, True, If(v["arg1_value"] == 7, If(v["arg2_dtype"] == 7, True, If(v["arg1_value"] == 8, If(v["arg2_dtype"] == 8, True, False), False)), False)), False)) if n else
          If(v["arg1_value"] == 6, If(v["arg2_dtype"] == 6, True, If(v["arg1_value"] == 7, If(v["arg2_dtype"] == 7, True, If(v["arg1_value"] == 8, If(v["arg2_dtype"] == 8, True, False), False)), False)), False))
)

def rule_12_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
