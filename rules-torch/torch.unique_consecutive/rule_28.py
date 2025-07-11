import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# combining rules for dim (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, v["arg2_value"] == -1, If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg2_value"] == -1, And((0 - v["arg1_ndim"]) <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"])))) if n else
          If(v["arg1_ndim"] == 0, v["arg2_value"] == -1, If(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg2_value"] == -1, And((0 - v["arg1_ndim"]) <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]))))
)

def rule_28_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 28
        rule_28(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
