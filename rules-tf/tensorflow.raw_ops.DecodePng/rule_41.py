import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If channels is not 0, it must be 1, 3 or 4. If tensor is rank 2 or higher, the final dimension has to align (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] != 0, And((Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 3), v["arg1_value"] == 4)), (If(v["arg2_ndim"] >= 2, (If(v["arg1_value"] == 1, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 1, If(v["arg1_value"] == 3, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3, If(v["arg1_value"] == 4, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 4, True)))), True))), True)) if n else
          If(v["arg1_value"] != 0, And((Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 3), v["arg1_value"] == 4)), (If(v["arg2_ndim"] >= 2, (If(v["arg1_value"] == 1, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 1, If(v["arg1_value"] == 3, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 3, If(v["arg1_value"] == 4, Select(v["arg2_shape"], v["arg2_ndim"] - 1) == 4, True)))), True))), True))
)

def rule_41_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg2_shape = Array('arg2_shape', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        for i in range(arg2.ndim):
            arg2_shape = Store(arg2_shape, i, arg2.shape[i])

        # Constraints for rule 41
        rule_41(solver, {'arg1_value': arg1_value, 'arg2_shape': arg2_shape, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_value': arg1['value'], 'arg2_shape': arg2['shape'], 'arg2_ndim': arg2['ndim']}, neg)
