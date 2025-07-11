import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If shape is an empty list, then memory_region_name must be none (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_length"] == 0, v["arg2_value"] == 6, False)) if n else
          If(v["arg1_length"] == 0, v["arg2_value"] == 6, False))
)

def rule_44_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 44
        rule_44(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
