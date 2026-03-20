import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The dtype should be one of the numerical dtypes (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11)) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11))
)

def rule_52_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 52
        rule_52(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_value': arg1['value']}, neg)
