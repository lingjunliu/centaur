import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If requires_grad is True, out should be a valid data type  (Rule 50)

rule_50 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], (Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 10), v["arg2_value"] == 11)), False)) if n else
          If(v["arg1_value"], (Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 7), v["arg2_value"] == 8), v["arg2_value"] == 10), v["arg2_value"] == 11)), False))
)

def rule_50_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 50
        rule_50(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_50(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
