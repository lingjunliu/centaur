import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the argument is a single-element tensor, its dtype should be such that it can be represented accurately by SymInt. (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(If(Or((v["arg1_ndim"] == 0), (And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 1))), (Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), (And(And(And(v["arg1_dtype"] == 6, (Select(v["arg1_range"], 0) % 1 == 0)), (Select(v["arg1_range"], 0) >= -2147483648)), (Select(v["arg1_range"], 1) <= 2147483647)))), (And(And(And(v["arg1_dtype"] == 7, (Select(v["arg1_range"], 0) % 1 == 0)), (Select(v["arg1_range"], 0) >= -2147483648)), (Select(v["arg1_range"], 1) <= 2147483647)))), (And(And(And(v["arg1_dtype"] == 8, (Select(v["arg1_range"], 0) % 1 == 0)), (Select(v["arg1_range"], 0) >= -2147483648)), (Select(v["arg1_range"], 1) <= 2147483647))))), True)) if n else
          If(Or((v["arg1_ndim"] == 0), (And(v["arg1_ndim"] == 1, Select(v["arg1_shape"], 0) == 1))), (Or(Or(Or(Or(Or(Or(Or(v["arg1_dtype"] == 1, v["arg1_dtype"] == 2), v["arg1_dtype"] == 3), v["arg1_dtype"] == 4), v["arg1_dtype"] == 5), (And(And(And(v["arg1_dtype"] == 6, (Select(v["arg1_range"], 0) % 1 == 0)), (Select(v["arg1_range"], 0) >= -2147483648)), (Select(v["arg1_range"], 1) <= 2147483647)))), (And(And(And(v["arg1_dtype"] == 7, (Select(v["arg1_range"], 0) % 1 == 0)), (Select(v["arg1_range"], 0) >= -2147483648)), (Select(v["arg1_range"], 1) <= 2147483647)))), (And(And(And(v["arg1_dtype"] == 8, (Select(v["arg1_range"], 0) % 1 == 0)), (Select(v["arg1_range"], 0) >= -2147483648)), (Select(v["arg1_range"], 1) <= 2147483647))))), True))
)

def rule_24_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 24
        rule_24(solver, {'arg1_range': arg1_range, 'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_range': arg1['range'], 'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
