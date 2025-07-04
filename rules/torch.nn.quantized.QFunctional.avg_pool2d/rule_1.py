import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# kernel_size and stride should be specified as either a single int or a tuple of two ints; padding should be a single int. (Rule 1)

rule_1 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg1_length"] == 2), (v["arg2_length"] == 2)), (v["arg3_value"] >= 0))) if n else
          And(And((v["arg1_length"] == 2), (v["arg2_length"] == 2)), (v["arg3_value"] >= 0)))
)

def rule_1_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 1
        rule_1(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_1(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
