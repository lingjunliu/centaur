import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Float v_1 multiplied by 2 must be greater than 10 (Rule 196)

rule_196 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] * 2 > 10) if n else
          v["arg1_value"] * 2 > 10)
)

def rule_196_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 196
        rule_196(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_196(solver, {'arg1_value': arg1['value']}, neg)
