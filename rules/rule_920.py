import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the tensor's number of dimension is equal to 1 or 0, and its data type is not bool, its max value can't be more than 10000. (Rule 920)

rule_920 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_ndim"] == 0, v["arg1_ndim"] == 1)), (v["arg1_dtype"] != 0)), Select(v["arg1_range"], 1) <= 10000, False)) if n else
          If(And((Or(v["arg1_ndim"] == 0, v["arg1_ndim"] == 1)), (v["arg1_dtype"] != 0)), Select(v["arg1_range"], 1) <= 10000, False))
)

def rule_920_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 920
        rule_920(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_920(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
