import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input is 1D and Target is 2D, or if input is 2D and Target is 1D then there is a shape incompatibility (Rule 121)

rule_121 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg1_ndim"] != v["arg2_ndim"]), (Or(v["arg1_ndim"] == 1, v["arg1_ndim"] == 2))), (Or(v["arg2_ndim"] == 1, v["arg2_ndim"] == 2)))) if n else
          And(And((v["arg1_ndim"] != v["arg2_ndim"]), (Or(v["arg1_ndim"] == 1, v["arg1_ndim"] == 2))), (Or(v["arg2_ndim"] == 1, v["arg2_ndim"] == 2))))
)

def rule_121_func(arg1, arg2, solver=None, neg=False):
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
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 121
        rule_121(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_121(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
