import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input is a float, its value must be representable by float32 (Rule 69)

rule_69 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] >= -3.4028235e38, v["arg1_value"] <= 3.4028235e38)) if n else
          And(v["arg1_value"] >= -3.4028235e38, v["arg1_value"] <= 3.4028235e38))
)

def rule_69_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 69
        rule_69(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_69(solver, {'arg1_value': arg1['value']}, neg)
