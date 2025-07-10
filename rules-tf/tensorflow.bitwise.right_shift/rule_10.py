import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# y values should be smaller than the bit width of x values (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 1, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 8) for i in range(6)]), If(v["arg1_dtype"] == 2, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 16) for i in range(6)]), If(v["arg1_dtype"] == 3, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 32) for i in range(6)]), If(v["arg1_dtype"] == 4, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 64) for i in range(6)]), If(v["arg1_dtype"] == 5, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 8) for i in range(6)]), If(v["arg1_dtype"] == 6, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 16) for i in range(6)]), False))))))) if n else
          If(v["arg1_dtype"] == 1, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 8) for i in range(6)]), If(v["arg1_dtype"] == 2, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 16) for i in range(6)]), If(v["arg1_dtype"] == 3, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 32) for i in range(6)]), If(v["arg1_dtype"] == 4, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 64) for i in range(6)]), If(v["arg1_dtype"] == 5, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 8) for i in range(6)]), If(v["arg1_dtype"] == 6, And([Implies(i < (v["arg2_ndim"] - 1 + 1), Select(v["arg2_range"], 1) < 16) for i in range(6)]), False)))))))
)

def rule_10_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 10
        rule_10(solver, {'arg1_dtype': arg1_dtype, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_dtype': arg1['dtype'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim']}, neg)
