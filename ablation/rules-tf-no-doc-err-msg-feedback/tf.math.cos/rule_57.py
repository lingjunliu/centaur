import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first element of tuple v_1 is True, then dtype of tensor v_2 should be either 7 or 8 (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg1_values"], 0), Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), True)) if n else
          If(Select(v["arg1_values"], 0), Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8), True))
)

def rule_57_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all(isinstance(e, bool) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), BoolSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 57
        rule_57(solver, {'arg1_values': arg1_values, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_values': arg1['values'], 'arg2_dtype': arg2['dtype']}, neg)
