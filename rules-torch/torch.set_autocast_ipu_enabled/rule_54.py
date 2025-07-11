import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# If autocast_ipu_enabled is true, scale_dtype cannot be an integer that's representative of int8, int16, int32, int64, or uint8 (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, (And(And(And(And(v["arg2_value"] != 1, v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5)), False)) if n else
          If(v["arg1_value"] == True, (And(And(And(And(v["arg2_value"] != 1, v["arg2_value"] != 2), v["arg2_value"] != 3), v["arg2_value"] != 4), v["arg2_value"] != 5)), False))
)

def rule_54_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 54
        rule_54(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
