import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If capacity and/or memory_limit are defined they have to be larger than -1 and smaller than 2147483647, if not both are 0 (Rule 65)

rule_65 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg1_value"] > -1, v["arg2_value"] > -1)), And((And(v["arg1_value"] > -1, v["arg1_value"] < 2147483647)), (And(v["arg2_value"] > -1, v["arg2_value"] < 2147483647))), And(v["arg1_value"] == 0, v["arg2_value"] == 0))) if n else
          If((Or(v["arg1_value"] > -1, v["arg2_value"] > -1)), And((And(v["arg1_value"] > -1, v["arg1_value"] < 2147483647)), (And(v["arg2_value"] > -1, v["arg2_value"] < 2147483647))), And(v["arg1_value"] == 0, v["arg2_value"] == 0)))
)

def rule_65_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 65
        rule_65(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_65(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
