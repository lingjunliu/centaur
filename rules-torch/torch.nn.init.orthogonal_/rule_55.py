import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Input tensor must have a supported dtype, must have at least 2 dimensions and must not be complex. (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(v["arg1_dtype"] != 6, v["arg1_dtype"] != 2), v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), v["arg1_ndim"] >= 2)) if n else
          And(And(And(And(v["arg1_dtype"] != 6, v["arg1_dtype"] != 2), v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), v["arg1_ndim"] >= 2))
)

def rule_55_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 55
        rule_55(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
