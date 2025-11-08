import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if v_1 and v_2 are both integers, v_1 must be larger than v_2 (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_value"] == int), (v["arg2_value"] == int)), v["arg1_value"] > v["arg2_value"], True)) if n else
          If(And((v["arg1_value"] == int), (v["arg2_value"] == int)), v["arg1_value"] > v["arg2_value"], True))
)

def rule_118_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 118
        rule_118(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
