import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Check that if theshold is very small we shouldn't use it as it can create instability. Threshold needs to be larger than Beta to ensure numeric properties. (Rule 144)

rule_144 = lambda s, v, n=False: (
    s.add(Not(And(v["arg2_value"] > 1e-7, v["arg1_value"] < v["arg2_value"])) if n else
          And(v["arg2_value"] > 1e-7, v["arg1_value"] < v["arg2_value"]))
)

def rule_144_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 144
        rule_144(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_144(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
