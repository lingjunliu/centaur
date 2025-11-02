import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 and v_2 are tensors representing matrices for multiplication, then number of dimensions can be either one or two but can't be 0 (Rule 83)

rule_83 = lambda s, v, n=False: (
    s.add(Not(And((Or(v["arg1_ndim"] == 1, v["arg1_ndim"] == 2)), (Or(v["arg2_ndim"] == 1, v["arg2_ndim"] == 2)))) if n else
          And((Or(v["arg1_ndim"] == 1, v["arg1_ndim"] == 2)), (Or(v["arg2_ndim"] == 1, v["arg2_ndim"] == 2))))
)

def rule_83_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 83
        rule_83(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_83(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
