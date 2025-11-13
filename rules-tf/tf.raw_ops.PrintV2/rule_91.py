import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The end should not be same as one of the accepted string value of tensors (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_dtype"] == 11, (And(And(v["arg1_value"] != 0, v["arg1_value"] != 1), v["arg1_value"] != 2)))) if n else
          And(v["arg2_dtype"] == 11, (And(And(v["arg1_value"] != 0, v["arg1_value"] != 1), v["arg1_value"] != 2))))
)

def rule_91_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 91
        rule_91(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
