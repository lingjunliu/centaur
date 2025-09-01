import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Should not be short as a output type, it also requires the same type between input and output (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_ndim"] > 0, And(And(v["arg1_dtype"] == v["arg2_dtype"], (If(v["arg1_dtype"] == 6, v["arg2_dtype"] != 8, True))), v["arg2_dtype"] != 2), True)) if n else
          If(v["arg2_ndim"] > 0, And(And(v["arg1_dtype"] == v["arg2_dtype"], (If(v["arg1_dtype"] == 6, v["arg2_dtype"] != 8, True))), v["arg2_dtype"] != 2), True))
)

def rule_118_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 118
        rule_118(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
