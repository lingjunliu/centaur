import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is a string and equal to channel_last or channel_first, then data type of tensor v_2 cannot be bool (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == 24, v["arg1_value"] == 25), v["arg2_dtype"] != 0, True)) if n else
          If(Or(v["arg1_value"] == 24, v["arg1_value"] == 25), v["arg2_dtype"] != 0, True))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 63
        rule_63(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
