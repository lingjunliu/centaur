import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Name must be a string if provided (Rule 11)

rule_11 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != False, True, True)) if n else
          If(v["arg1_value"] != False, True, True))
)

def rule_11_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str) or isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 11
        rule_11(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_11(solver, {'arg1_value': arg1['value']}, neg)
