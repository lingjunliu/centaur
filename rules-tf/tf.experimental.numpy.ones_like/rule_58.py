import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the user requests a specific dtype and not a dtype object, the integer representation must not correspond to the non-supported torch int64 dtype. (Rule 58)

rule_58 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 12, v["arg1_value"] != 4, True)) if n else
          If(v["arg1_value"] < 12, v["arg1_value"] != 4, True))
)

def rule_58_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 58
        rule_58(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_58(solver, {'arg1_value': arg1['value']}, neg)
