import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If requires_grad is specified as True, then dtype cannot be complex if dtype is not none (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, (If(v["arg2_value"] == 12, True, (And(v["arg2_value"] != 9, v["arg2_value"] != 10)))), True)) if n else
          If(v["arg1_value"] == True, (If(v["arg2_value"] == 12, True, (And(v["arg2_value"] != 9, v["arg2_value"] != 10)))), True))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not ((isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)) or (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
