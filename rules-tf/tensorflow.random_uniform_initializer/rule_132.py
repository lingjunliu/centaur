import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# either minval is not greater than maxval, or both are infinite (Rule 132)

rule_132 = lambda s, v, n=False: (
    s.add(Not(Or(Or(v["arg1_value"] <= v["arg2_value"], (And(v["arg1_value"] == 3.4028235e+38, v["arg2_value"] == 3.4028235e+38))), (And(v["arg1_value"] == -3.4028235e+38, v["arg2_value"] == -3.4028235e+38)))) if n else
          Or(Or(v["arg1_value"] <= v["arg2_value"], (And(v["arg1_value"] == 3.4028235e+38, v["arg2_value"] == 3.4028235e+38))), (And(v["arg1_value"] == -3.4028235e+38, v["arg2_value"] == -3.4028235e+38))))
)

def rule_132_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)

        # Constraints for rule 132
        rule_132(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_132(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
