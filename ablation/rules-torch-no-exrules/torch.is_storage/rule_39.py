import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Object must be a tensor with floating point dtype and at least one element (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or((v["arg1_dtype"] == 6), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8))), (v["arg1_ndim"] > 0))) if n else
          And((Or(Or((v["arg1_dtype"] == 6), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8))), (v["arg1_ndim"] > 0)))
)

def rule_39_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 39
        rule_39(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
