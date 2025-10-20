import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If 'a' and 'x' are both tensors, their dtypes must be float32 or float64 and the same and must have less than 5 dimensions (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_ndim"] < 5), v["arg2_ndim"] < 5)) if n else
          And(And(And(And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_ndim"] < 5), v["arg2_ndim"] < 5))
)

def rule_46_func(arg1, arg2, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
