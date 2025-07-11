import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If k is a tuple, k must be int32 representable and k[0] must be less than k[1] (Rule 95)

rule_95 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(Select(v["arg1_values"], 0) >= -2147483648, Select(v["arg1_values"], 0) <= 2147483647), Select(v["arg1_values"], 1) >= -2147483648), Select(v["arg1_values"], 1) <= 2147483647), Select(v["arg1_values"], 0) <= Select(v["arg1_values"], 1))) if n else
          And(And(And(And(Select(v["arg1_values"], 0) >= -2147483648, Select(v["arg1_values"], 0) <= 2147483647), Select(v["arg1_values"], 1) >= -2147483648), Select(v["arg1_values"], 1) <= 2147483647), Select(v["arg1_values"], 0) <= Select(v["arg1_values"], 1)))
)

def rule_95_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 95
        rule_95(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_95(solver, {'arg1_values': arg1['values']}, neg)
