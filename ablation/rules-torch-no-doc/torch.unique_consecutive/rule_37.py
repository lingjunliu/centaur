import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dimension is None and counts is not None, tensor must be int and counts must be int. (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] == 6, v["arg3_ndim"] > 0), (And(And(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5), v["arg3_dtype"] >= 1), v["arg3_dtype"] <= 5)), True)) if n else
          If(And(v["arg2_value"] == 6, v["arg3_ndim"] > 0), (And(And(And(v["arg1_dtype"] >= 1, v["arg1_dtype"] <= 5), v["arg3_dtype"] >= 1), v["arg3_dtype"] <= 5)), True))
)

def rule_37_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, str) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 37
        rule_37(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
