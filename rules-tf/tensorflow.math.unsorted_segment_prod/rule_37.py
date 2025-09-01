import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Final and Corrected Combined Constraints (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 13), v["arg1_dtype"] == 6), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 10), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), (Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4))), v["arg3_ndim"] == 0), Select(v["arg3_range"], 0) >= 0), And([Implies(i < (If(v["arg2_ndim"] < 0, 0, v["arg2_ndim"] - 1) + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg2_range"], 1) < Select(v["arg3_range"], 1))) for i in range(6)]))) if n else
          And(And(And(And(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8), v["arg1_dtype"] == 3), v["arg1_dtype"] == 5), v["arg1_dtype"] == 2), v["arg1_dtype"] == 1), v["arg1_dtype"] == 9), v["arg1_dtype"] == 4), v["arg1_dtype"] == 13), v["arg1_dtype"] == 6), v["arg1_dtype"] == 15), v["arg1_dtype"] == 16), v["arg1_dtype"] == 10), v["arg1_dtype"] == 17), v["arg1_dtype"] == 18)), (Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4))), (Or(v["arg3_dtype"] == 3, v["arg3_dtype"] == 4))), v["arg3_ndim"] == 0), Select(v["arg3_range"], 0) >= 0), And([Implies(i < (If(v["arg2_ndim"] < 0, 0, v["arg2_ndim"] - 1) + 1), And(Select(v["arg1_shape"], i) == Select(v["arg2_shape"], i), Select(v["arg2_range"], 1) < Select(v["arg3_range"], 1))) for i in range(6)])))
)

def rule_37_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 37
        rule_37(solver, {'arg1_shape': arg1_shape, 'arg1_dtype': arg1_dtype, 'arg2_shape': arg2_shape, 'arg2_range': arg2_range, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_range': arg3_range, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_shape': arg1['shape'], 'arg1_dtype': arg1['dtype'], 'arg2_shape': arg2['shape'], 'arg2_range': arg2['range'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_range': arg3['range'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
