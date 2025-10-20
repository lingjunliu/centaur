import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Ensure concat dim is an integer (Rule 32)

rule_32 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or((Or([And(i < (5 + 1), v["arg1_value"] == i) for i in range(6)])), (Or([And(i < (8 + 1), v["arg1_value"] == i) for i in range(6)]))), v["arg1_value"] == 0), (Or([And(i < (10 + 1), v["arg1_value"] == i) for i in range(6)])))) if n else
          Or(Or(Or((Or([And(i < (5 + 1), v["arg1_value"] == i) for i in range(6)])), (Or([And(i < (8 + 1), v["arg1_value"] == i) for i in range(6)]))), v["arg1_value"] == 0), (Or([And(i < (10 + 1), v["arg1_value"] == i) for i in range(6)]))))
)

def rule_32_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 32
        rule_32(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_32(solver, {'arg1_value': arg1['value']}, neg)
