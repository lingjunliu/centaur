import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if the first element in tuple of int > 0, then the second element should be positive (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 1) > 0, False)) if n else
          If(Select(v["arg1_values"], 0) > 0, Select(v["arg1_values"], 1) > 0, False))
)

def rule_60_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 60
        rule_60(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_values': arg1['values']}, neg)
