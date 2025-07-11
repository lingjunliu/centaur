import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dtype is specified, it must be a floating or complex number data type, indicated by values 6-10 or it can be None indicating it will be determined from the context (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(Or((v["arg1_value"] == -1), (And(6 <= v["arg1_value"], v["arg1_value"] <= 10)))) if n else
          Or((v["arg1_value"] == -1), (And(6 <= v["arg1_value"], v["arg1_value"] <= 10))))
)

def rule_64_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 64
        rule_64(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_value': arg1['value']}, neg)
