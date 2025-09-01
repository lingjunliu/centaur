import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input and mat2 are of different dtypes, the output dtype cannot be an integer (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] != v["arg2_dtype"], If(v["arg3_ndim"] > 0, (And(And(And(And(v["arg3_dtype"] != 1, v["arg3_dtype"] != 2), v["arg3_dtype"] != 3), v["arg3_dtype"] != 4), v["arg3_dtype"] != 5)), True), True)) if n else
          If(v["arg1_dtype"] != v["arg2_dtype"], If(v["arg3_ndim"] > 0, (And(And(And(And(v["arg3_dtype"] != 1, v["arg3_dtype"] != 2), v["arg3_dtype"] != 3), v["arg3_dtype"] != 4), v["arg3_dtype"] != 5)), True), True))
)

def rule_39_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 39
        rule_39(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
