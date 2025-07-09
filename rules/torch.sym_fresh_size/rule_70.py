import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# tensor should have at least two dimensions and first and second dimension sum is more than 10, check for max shape less than 20 or max function returns greater than 100 (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(Or((And(And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) > 10), Select(v["arg1_shape"], 0) < 20), Select(v["arg1_shape"], 1) < 20)), Select(v["arg1_range"], 1) > 100)) if n else
          Or((And(And(And(v["arg1_ndim"] >= 2, Select(v["arg1_shape"], 0) + Select(v["arg1_shape"], 1) > 10), Select(v["arg1_shape"], 0) < 20), Select(v["arg1_shape"], 1) < 20)), Select(v["arg1_range"], 1) > 100))
)

def rule_70_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 70
        rule_70(solver, {'arg1_ndim': arg1_ndim, 'arg1_range': arg1_range, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_ndim': arg1['ndim'], 'arg1_range': arg1['range'], 'arg1_shape': arg1['shape']}, neg)
