import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The data type of 'out' must be large enough to hold the result of asinh(input (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 2, And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), If(v["arg1_dtype"] == 7, (And(Select(v["arg1_range"], 0) > -7.6293945E4, Select(v["arg1_range"], 1) < 7.6293945E4)), (And(Select(v["arg1_range"], 0) > -7.6293945E4, Select(v["arg1_range"], 1) < 7.6293945E4)))) for i in range(6)])), True)) if n else
          If(v["arg2_dtype"] == 2, And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), And([Implies(i < (If(v["arg1_ndim"] > 0, v["arg1_ndim"] - 1, 0) + 1), If(v["arg1_dtype"] == 7, (And(Select(v["arg1_range"], 0) > -7.6293945E4, Select(v["arg1_range"], 1) < 7.6293945E4)), (And(Select(v["arg1_range"], 0) > -7.6293945E4, Select(v["arg1_range"], 1) < 7.6293945E4)))) for i in range(6)])), True))
)

def rule_13_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 13
        rule_13(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
