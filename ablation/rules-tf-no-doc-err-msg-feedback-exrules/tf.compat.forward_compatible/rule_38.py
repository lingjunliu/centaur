import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is an integer and v_2 is a float, then v_2 should be in range [-v_1, v_1] (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_value"] >= (0 - v["arg1_value"]), v["arg2_value"] <= v["arg1_value"])) if n else
          And(v["arg2_value"] >= (0 - v["arg1_value"]), v["arg2_value"] <= v["arg1_value"]))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
