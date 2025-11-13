import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# output_idx_type should be int32 or int64 if and only if a value is passed (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), True, If(And(v["arg1_value"] != 3, v["arg1_value"] != 4), True, False))) if n else
          If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), True, If(And(v["arg1_value"] != 3, v["arg1_value"] != 4), True, False)))
)

def rule_13_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 13
        rule_13(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_value': arg1['value']}, neg)
