import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Dtype should be None, float16, float32, float64, complex64, complex128 (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or((v["arg1_value"] == 12), (v["arg1_value"] == 6)), (v["arg1_value"] == 7)), (v["arg1_value"] == 8)), (v["arg1_value"] == 9)), (v["arg1_value"] == 10))) if n else
          Or(Or(Or(Or(Or((v["arg1_value"] == 12), (v["arg1_value"] == 6)), (v["arg1_value"] == 7)), (v["arg1_value"] == 8)), (v["arg1_value"] == 9)), (v["arg1_value"] == 10)))
)

def rule_78_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 78
        rule_78(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_value': arg1['value']}, neg)
