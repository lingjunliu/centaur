import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If atol or rtol is tensor, then the other must also be a tensor with correct type and shape (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If((v["arg2_ndim"] > 0), (And(And(And(v["arg3_ndim"] > 0, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))), v["arg3_ndim"] <= v["arg1_ndim"] - 2)), If((v["arg3_ndim"] > 0), (And(And(And(v["arg2_ndim"] > 0, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))), v["arg2_ndim"] <= v["arg1_ndim"] - 2)), True))) if n else
          If((v["arg2_ndim"] > 0), (And(And(And(v["arg3_ndim"] > 0, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))), v["arg3_ndim"] <= v["arg1_ndim"] - 2)), If((v["arg3_ndim"] > 0), (And(And(And(v["arg2_ndim"] > 0, (Or(v["arg2_dtype"] == 7, v["arg2_dtype"] == 8))), (Or(v["arg3_dtype"] == 7, v["arg3_dtype"] == 8))), v["arg2_ndim"] <= v["arg1_ndim"] - 2)), True)))
)

def rule_45_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
