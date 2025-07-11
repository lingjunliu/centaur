import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The length of a list has to be smaller than a number and the number has to be positive (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] < v["arg2_value"], v["arg2_value"] > 0)) if n else
          And(v["arg1_length"] < v["arg2_value"], v["arg2_value"] > 0))
)

def rule_64_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 64
        rule_64(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
