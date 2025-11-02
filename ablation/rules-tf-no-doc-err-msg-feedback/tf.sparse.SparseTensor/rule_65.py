import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Either integer v_1 should be greater than 0 or it must be less than the length of list v_2. (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"] > 0, v["arg1_value"] < v["arg2_length"])) if n else
          Or(v["arg1_value"] > 0, v["arg1_value"] < v["arg2_length"]))
)

def rule_65_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
