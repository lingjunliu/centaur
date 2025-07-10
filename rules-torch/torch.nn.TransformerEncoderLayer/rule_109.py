import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# layer_norm_eps must be a small positive number, within (1e-06, 0.001 (Rule 109)

rule_109 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] > 1e-6, v["arg1_value"] < 0.001), v["arg1_value"] < v["arg2_value"]), (Or(v["arg3_value"] == True, v["arg3_value"] == False)))) if n else
          And(And(And(v["arg1_value"] > 1e-6, v["arg1_value"] < 0.001), v["arg1_value"] < v["arg2_value"]), (Or(v["arg3_value"] == True, v["arg3_value"] == False))))
)

def rule_109_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Real('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 109
        rule_109(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
