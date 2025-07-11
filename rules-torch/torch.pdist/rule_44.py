import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# the input must be a 2D tensor AND either it's not Half, or p is >= 0 (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_ndim"] == 2, (Or(v["arg1_dtype"] != 6, v["arg2_value"] >= 0)))) if n else
          And(v["arg1_ndim"] == 2, (Or(v["arg1_dtype"] != 6, v["arg2_value"] >= 0))))
)

def rule_44_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 44
        rule_44(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
