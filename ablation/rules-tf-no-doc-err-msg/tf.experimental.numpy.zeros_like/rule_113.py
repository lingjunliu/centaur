import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ensure dtype value is not a combination that's invalid (e.g. bool combined with complex (Rule 113)

rule_113 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 9), v["arg1_value"] == 10), False, True)) if n else
          If(Or(Or(v["arg1_value"] == 0, v["arg1_value"] == 9), v["arg1_value"] == 10), False, True))
)

def rule_113_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 113
        rule_113(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_113(solver, {'arg1_value': arg1['value']}, neg)
