import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The list of string cardinality should have at least one element if it isn't known (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == -1, v["arg1_length"] > 0, False)) if n else
          If(v["arg2_value"] == -1, v["arg1_length"] > 0, False))
)

def rule_55_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
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

        # Constraints for rule 55
        rule_55(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
