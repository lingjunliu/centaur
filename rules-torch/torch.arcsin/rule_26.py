import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if the output tensor is specified to be Short, ensure that any potential NaN or inf values resulting from arcsin on values outside [-1, 1] in the input, will not cause a cast exception (Rule 26)

rule_26 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_dtype"] == 2, And([Implies(i < (v["arg1_ndim"] - 1 + 1), And((Select(v["arg1_range"], 0) >= -1), (Select(v["arg1_range"], 1) <= 1))) for i in range(6)]), True)) if n else
          If(v["arg2_dtype"] == 2, And([Implies(i < (v["arg1_ndim"] - 1 + 1), And((Select(v["arg1_range"], 0) >= -1), (Select(v["arg1_range"], 1) <= 1))) for i in range(6)]), True))
)

def rule_26_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 26
        rule_26(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_26(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype']}, neg)
