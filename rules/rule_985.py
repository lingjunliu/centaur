import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the number of dimensions of tensor is 2, and the string value is not "none", then the minimum value of the tensor must be less or equal than 100 (Rule 985)

rule_985 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] == 2, v["arg2_value"] != 6), Select(v["arg1_range"], 0) <= 100, False)) if n else
          If(And(v["arg1_ndim"] == 2, v["arg2_value"] != 6), Select(v["arg1_range"], 0) <= 100, False))
)

def rule_985_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 985
        rule_985(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_985(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg2_value': arg2['value']}, neg)
