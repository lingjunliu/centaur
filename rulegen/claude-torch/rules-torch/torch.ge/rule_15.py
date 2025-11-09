import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# output tensor must not have complex dtype (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg3_dtype"] != 9, v["arg3_dtype"] != 10), v["arg1_ndim"] >= 0), v["arg2_ndim"] >= 0)) if n else
          And(And(And(v["arg3_dtype"] != 9, v["arg3_dtype"] != 10), v["arg1_ndim"] >= 0), v["arg2_ndim"] >= 0))
)

def rule_15_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 15
        rule_15(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
