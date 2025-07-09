import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# first element of a tuple must be greater than a number or equal to it or is negative and tuple size is 3 or tensor dimension is 1 and integer is positive  (Rule 92)

rule_92 = lambda s, v, n=False: (
    s.add(Not(Or((And((Or(Select(v["arg1_values"], 0) >= v["arg2_value"], Select(v["arg1_values"], 0) < 0)), v["arg1_length"] == 3)), (And(v["arg3_ndim"] == 1, v["arg2_value"] > 0)))) if n else
          Or((And((Or(Select(v["arg1_values"], 0) >= v["arg2_value"], Select(v["arg1_values"], 0) < 0)), v["arg1_length"] == 3)), (And(v["arg3_ndim"] == 1, v["arg2_value"] > 0))))
)

def rule_92_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 92
        rule_92(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_92(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
