import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input tuple has a length of 2 and the input string is same, tensor dtype must be int (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] == 2, v["arg2_value"] == 22), Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), True)) if n else
          If(And(v["arg1_length"] == 2, v["arg2_value"] == 22), Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), True))
)

def rule_100_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = String('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 100
        rule_100(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
