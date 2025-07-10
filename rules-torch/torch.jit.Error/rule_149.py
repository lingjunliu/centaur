import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If tensor v_1 and tensor v_2 are both defined, the number of dimensions has to be the same (Rule 149)

rule_149 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_ndim"] >= 0, v["arg2_ndim"] >= 0), v["arg1_ndim"] == v["arg2_ndim"], False)) if n else
          If(And(v["arg1_ndim"] >= 0, v["arg2_ndim"] >= 0), v["arg1_ndim"] == v["arg2_ndim"], False))
)

def rule_149_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 149
        rule_149(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_149(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim']}, neg)
