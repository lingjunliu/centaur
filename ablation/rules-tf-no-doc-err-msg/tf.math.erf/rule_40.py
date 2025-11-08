import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If string v_1 is equal to 'channels_first', then integer v_2 should be equal to 1; otherwise, if string v_1 is equal to 'channels_last', then integer v_2 should be equal to 3. (Rule 40)

rule_40 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 25, v["arg2_value"] == 1, If(v["arg1_value"] == 24, v["arg2_value"] == 3, True))) if n else
          If(v["arg1_value"] == 25, v["arg2_value"] == 1, If(v["arg1_value"] == 24, v["arg2_value"] == 3, True)))
)

def rule_40_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 40
        rule_40(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_40(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
