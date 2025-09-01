import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the input is complex and dim is a list(int (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), And(Select(v["arg2_values"], 0) >= (0 - v["arg1_ndim"]), Select(v["arg2_values"], 0) < v["arg1_ndim"]), True)) if n else
          If(Or(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), And(Select(v["arg2_values"], 0) >= (0 - v["arg1_ndim"]), Select(v["arg2_values"], 0) < v["arg1_ndim"]), True))
)

def rule_71_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 71
        rule_71(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_values': arg2['values']}, neg)
