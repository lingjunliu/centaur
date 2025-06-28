import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If tensor v_1 has number of dimension equal to 4 and maximum element is negative, all its shapes must be divisible by 2 (Rule 794)

rule_794 = lambda s, v, n=False: (
    s.add(Not(If(And((v["arg1_ndim"] == 4), (Select(v["arg1_range"], 1) < 0)), And(And(And((Select(v["arg1_shape"], 0) / 2 == (Select(v["arg1_shape"], 0) + 1 - 1) / 2), (Select(v["arg1_shape"], 1) / 2 == (Select(v["arg1_shape"], 1) + 1 - 1) / 2)), (Select(v["arg1_shape"], 2) / 2 == (Select(v["arg1_shape"], 2) + 1 - 1) / 2)), (Select(v["arg1_shape"], 3) / 2 == (Select(v["arg1_shape"], 3) + 1 - 1) / 2)), False)) if n else
          If(And((v["arg1_ndim"] == 4), (Select(v["arg1_range"], 1) < 0)), And(And(And((Select(v["arg1_shape"], 0) / 2 == (Select(v["arg1_shape"], 0) + 1 - 1) / 2), (Select(v["arg1_shape"], 1) / 2 == (Select(v["arg1_shape"], 1) + 1 - 1) / 2)), (Select(v["arg1_shape"], 2) / 2 == (Select(v["arg1_shape"], 2) + 1 - 1) / 2)), (Select(v["arg1_shape"], 3) / 2 == (Select(v["arg1_shape"], 3) + 1 - 1) / 2)), False))
)

def rule_794_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 794
        rule_794(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_shape': arg1_shape})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_794(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_shape': arg1['shape']}, neg)
