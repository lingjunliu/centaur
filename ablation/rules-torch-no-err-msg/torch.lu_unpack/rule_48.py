import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If LU_data has dim > 2 and unpack_pivots = true, then LU_pivots should have integer type (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] > 2, v["arg3_value"] == True), (Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5)), True)) if n else
          If(And(v["arg1_ndim"] > 2, v["arg3_value"] == True), (Or(Or(Or(Or(v["arg2_dtype"] == 1, v["arg2_dtype"] == 2), v["arg2_dtype"] == 3), v["arg2_dtype"] == 4), v["arg2_dtype"] == 5)), True))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)

        # Constraints for rule 48
        rule_48(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_value': arg3['value']}, neg)
