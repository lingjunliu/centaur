import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The central_fraction is not a list (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] != Select(v["arg2_values"], 0)) if n else
          v["arg1_value"] != Select(v["arg2_values"], 0))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, (float, np.floating)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), RealSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 38
        rule_38(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values']}, neg)
