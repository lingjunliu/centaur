import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check num_bits is integer type and is in range 2-16 inclusive (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(Or([And(i < (5 + 1), v["arg1_value"] == i) for i in range(6)]), And(v["arg1_value"] >= 2, v["arg1_value"] <= 16), False)) if n else
          If(Or([And(i < (5 + 1), v["arg1_value"] == i) for i in range(6)]), And(v["arg1_value"] >= 2, v["arg1_value"] <= 16), False))
)

def rule_49_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 49
        rule_49(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_value': arg1['value']}, neg)
