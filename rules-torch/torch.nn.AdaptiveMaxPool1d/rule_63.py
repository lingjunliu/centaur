import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check for storage overflow in adaptive_max_pool1d based on both input dimension and output size. If the non-batch dimensions are zero, it's fine. (Rule 63)

rule_63 = lambda s, v, n=False: (
    s.add(Not(If(Or((And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 1) == 0)), (And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 2) == 0))), True, Or((And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) < 2000000000 / v["arg2_value"])), (And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 2000000000 / v["arg2_value"]))))) if n else
          If(Or((And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 1) == 0)), (And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 2) == 0))), True, Or((And(v["arg1_ndim"] == 2, Select(v["arg1_shape"], 0) < 2000000000 / v["arg2_value"])), (And(v["arg1_ndim"] == 3, Select(v["arg1_shape"], 0) * Select(v["arg1_shape"], 1) < 2000000000 / v["arg2_value"])))))
)

def rule_63_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_shape = Array('arg1_shape', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        for i in range(arg1.ndim):
            arg1_shape = Store(arg1_shape, i, arg1.shape[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 63
        rule_63(solver, {'arg1_shape': arg1_shape, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_63(solver, {'arg1_shape': arg1['shape'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
