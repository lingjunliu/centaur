import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the dtype is an integer, it must be between 1 and 5. Otherwise, it has to be 0 (Rule 89)

rule_89 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] >= 1, v["arg1_value"] <= 5), True, If(v["arg1_value"] == 0, True, False))) if n else
          If(And(v["arg1_value"] >= 1, v["arg1_value"] <= 5), True, If(v["arg1_value"] == 0, True, False)))
)

def rule_89_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))

        # Constraints for rule 89
        rule_89(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_89(solver, {'arg1_value': arg1['value']}, neg)
