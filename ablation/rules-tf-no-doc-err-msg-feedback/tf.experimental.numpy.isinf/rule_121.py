import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If 1st element in tuple is greater than 1, Dtype in tensor should be 7 or 8 (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(If(Select(v["arg2_values"], 0) > 1, Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), True)) if n else
          If(Select(v["arg2_values"], 0) > 1, Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), True))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 121
        rule_121(solver, {'arg1_dtype': arg1_dtype, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_dtype': arg1['dtype'], 'arg2_values': arg2['values']}, neg)
