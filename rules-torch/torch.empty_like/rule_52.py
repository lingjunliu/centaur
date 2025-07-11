import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# memory_format must be of dtype or memory_format type (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] != 11) if n else
          v["arg1_value"] != 11)
)

def rule_52_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)) or isinstance(arg1, str)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value']}, neg)
