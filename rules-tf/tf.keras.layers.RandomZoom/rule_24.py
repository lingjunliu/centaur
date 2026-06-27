import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# height_factor, width_factor, and data_format must have valid types and values (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_value"] >= -1.0, v["arg1_value"] <= 1.0), v["arg2_value"] >= -1.0), v["arg2_value"] <= 1.0), (Or(v["arg3_value"] == 24, v["arg3_value"] == 25)))) if n else
          And(And(And(And(v["arg1_value"] >= -1.0, v["arg1_value"] <= 1.0), v["arg2_value"] >= -1.0), v["arg2_value"] <= 1.0), (Or(v["arg3_value"] == 24, v["arg3_value"] == 25))))
)

def rule_24_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 24
        rule_24(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
