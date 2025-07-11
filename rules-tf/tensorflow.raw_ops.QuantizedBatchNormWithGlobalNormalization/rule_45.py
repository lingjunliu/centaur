import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if m, v, beta, gamma are not provided then t, m, v, beta and gamma can be any value. (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(v["arg2_ndim"] != 1, v["arg3_ndim"] != 1), v["arg4_ndim"] != 1), v["arg5_ndim"] != 1), True, False)) if n else
          If(Or(Or(Or(v["arg2_ndim"] != 1, v["arg3_ndim"] != 1), v["arg4_ndim"] != 1), v["arg5_ndim"] != 1), True, False))
)

def rule_45_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, np.ndarray):
            return False
        if not isinstance(arg5, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_ndim = Int('arg4_ndim')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 45
        rule_45(solver, {'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim'], 'arg5_ndim': arg5['ndim']}, neg)
