import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Combine dtype, checks on magnitude, close to infinity, dimensions (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 13)), (And(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), v["arg1_dtype"] != 12))), (If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), (And(Select(v["arg1_range"], 0) > -1e5, Select(v["arg1_range"], 1) < 1e5)), True))), (If(v["arg1_dtype"] == 7, And(Select(v["arg1_range"], 0) > -3.4e38, Select(v["arg1_range"], 1) < 3.4e38), If(v["arg1_dtype"] == 8, And(Select(v["arg1_range"], 0) > -1.7e308, Select(v["arg1_range"], 1) < 1.7e308), True)))), (Or(v["arg1_ndim"] > 0, (v["arg1_ndim"] == 0))))) if n else
          And(And(And(And((Or(Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10), v["arg1_dtype"] == 13)), (And(And(And(And(And(And(v["arg1_dtype"] != 0, v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), v["arg1_dtype"] != 12))), (If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), (And(Select(v["arg1_range"], 0) > -1e5, Select(v["arg1_range"], 1) < 1e5)), True))), (If(v["arg1_dtype"] == 7, And(Select(v["arg1_range"], 0) > -3.4e38, Select(v["arg1_range"], 1) < 3.4e38), If(v["arg1_dtype"] == 8, And(Select(v["arg1_range"], 0) > -1.7e308, Select(v["arg1_range"], 1) < 1.7e308), True)))), (Or(v["arg1_ndim"] > 0, (v["arg1_ndim"] == 0)))))
)

def rule_98_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 98
        rule_98(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)
