import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# First tensor is of type float and min is less than 10, the second tensor must have at least one dimension and dtype must be smaller than 10 or equal to 12 (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_dtype"] >= 7, v["arg1_dtype"] <= 9), Select(v["arg1_range"], 0) < 10), And(v["arg2_ndim"] >= 1, (Or(v["arg2_dtype"] < 10, v["arg2_dtype"] == 12))), True)) if n else
          If(And(And(v["arg1_dtype"] >= 7, v["arg1_dtype"] <= 9), Select(v["arg1_range"], 0) < 10), And(v["arg2_ndim"] >= 1, (Or(v["arg2_dtype"] < 10, v["arg2_dtype"] == 12))), True))
)

def rule_61_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 61
        rule_61(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
