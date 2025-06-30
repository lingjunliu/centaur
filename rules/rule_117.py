import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# if the tensor's dtype is integer type and dimension > 1, and max value is less than or equal to 10, then min value must be greater than 0 (Rule 117)

rule_117 = lambda s, v, n=False: (
    s.add(Not(If(And(And((Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), v["arg1_ndim"] > 1), Select(v["arg1_range"], 1) <= 10), Select(v["arg1_range"], 0) >= 0, False)) if n else
          If(And(And((Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5)), v["arg1_ndim"] > 1), Select(v["arg1_range"], 1) <= 10), Select(v["arg1_range"], 0) >= 0, False))
)

def rule_117_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 117
        rule_117(solver, {'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_117(solver, {'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim']}, neg)
