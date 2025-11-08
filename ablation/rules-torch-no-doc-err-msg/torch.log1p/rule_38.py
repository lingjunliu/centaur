import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The type of data has to be either bool, int, or float (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] == 0, (And(v["arg1_value"] >= 1, v["arg1_value"] <= 9)))) if n else
          Or(v["arg1_value"] == 0, (And(v["arg1_value"] >= 1, v["arg1_value"] <= 9))))
)

def rule_38_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value']}, neg)
