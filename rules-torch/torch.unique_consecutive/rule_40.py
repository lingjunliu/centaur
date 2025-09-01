import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# return_inverse and return_counts should both have boolean type (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, If(v["arg2_value"] == True, True, If(v["arg2_value"] == False, True, False)), If(v["arg1_value"] == False, If(v["arg2_value"] == True, True, If(v["arg2_value"] == False, True, False)), False))) if n else
          If(v["arg1_value"] == True, If(v["arg2_value"] == True, True, If(v["arg2_value"] == False, True, False)), If(v["arg1_value"] == False, If(v["arg2_value"] == True, True, If(v["arg2_value"] == False, True, False)), False)))
)

def rule_40_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, bool) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating)) or isinstance(arg1, str) or (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType))):
            return False
        if not (isinstance(arg2, bool) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating)) or isinstance(arg2, str) or (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
