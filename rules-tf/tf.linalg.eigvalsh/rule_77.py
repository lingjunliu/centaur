import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# tensor's dtype must be one of the allowed values to avoid InvalidArgumentError and NotFoundError (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or((And(v["arg1_ndim"] > 1, (Or((Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Select(v["arg1_shape"], 0) == 0))))), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) > -1)), (If(v["arg1_ndim"] == 1, True, False))), (1 < 2)), (v["arg1_dtype"] == 12)), (Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1))), (1 < 10))) if n else
          Or(Or(Or(Or(Or(Or((And(v["arg1_ndim"] > 1, (Or((Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10)), (Select(v["arg1_shape"], 0) == 0))))), (Select(v["arg1_shape"], v["arg1_ndim"] - 1) > -1)), (If(v["arg1_ndim"] == 1, True, False))), (1 < 2)), (v["arg1_dtype"] == 12)), (Select(v["arg1_range"], 0) < Select(v["arg1_range"], 1))), (1 < 10)))
)

def rule_77_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 77
        rule_77(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_shape': arg1_shape, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_shape': arg1['shape'], 'arg1_range': arg1['range']}, neg)
