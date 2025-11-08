import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is tuple of floats and v_2 is an integer then v_2 must be less than the product of the floats (Rule 52)

rule_52 = lambda s, v, n=False: (
    s.add(Not(v["arg2_value"] < Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1)) if n else
          v["arg2_value"] < Select(v["arg1_values"], 0) * Select(v["arg1_values"], 1))
)

def rule_52_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, (float, np.floating)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), RealSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 52
        rule_52(solver, {'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_52(solver, {'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)
