import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If auto cast IPU enabled is true and scale_dtype is an int, the value has to be 0 or one of the float representations. If scale_dtype is a dtype, it has to be float representation (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, (Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), (Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8)))), False)) if n else
          If(v["arg1_value"] == True, (Or(Or(Or(Or(v["arg2_value"] == 0, v["arg2_value"] == 6), v["arg2_value"] == 7), v["arg2_value"] == 8), (Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8)))), False))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
