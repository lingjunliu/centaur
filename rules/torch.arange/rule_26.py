import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# start, end, and step must all be ints or all be floats (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(And(And((v["arg1_value"] > 0), (v["arg2_value"] > 0)), (v["arg3_value"] > 0)), True, If(And(And((v["arg1_value"] == 0), (v["arg2_value"] == 0)), (v["arg3_value"] == 0)), True, If(And(And((v["arg1_value"] < 0), (v["arg2_value"] < 0)), (v["arg3_value"] < 0)), True, (If(Or((v["arg1_value"] > 0), (v["arg1_value"] < 0)), (If(Or((v["arg2_value"] > 0), (v["arg2_value"] < 0)), (If(Or((v["arg3_value"] > 0), (v["arg3_value"] < 0)), True, False)), False)), False)))))) if n else
          If(And(And((v["arg1_value"] > 0), (v["arg2_value"] > 0)), (v["arg3_value"] > 0)), True, If(And(And((v["arg1_value"] == 0), (v["arg2_value"] == 0)), (v["arg3_value"] == 0)), True, If(And(And((v["arg1_value"] < 0), (v["arg2_value"] < 0)), (v["arg3_value"] < 0)), True, (If(Or((v["arg1_value"] > 0), (v["arg1_value"] < 0)), (If(Or((v["arg2_value"] > 0), (v["arg2_value"] < 0)), (If(Or((v["arg3_value"] > 0), (v["arg3_value"] < 0)), True, False)), False)), False))))))
)

def rule_26_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not ((isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)) or isinstance(arg3, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 26
        rule_26(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
