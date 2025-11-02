import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check if means, stddevs, minvals, and maxvals are tensors when their ndim > 0. (Rule 100)

rule_100 = lambda s, v, n=False: (
    s.add(Not(And(And(And((If(v["arg1_ndim"] > 0, True, True)), (If(v["arg2_ndim"] > 0, True, True))), (If(v["arg3_ndim"] > 0, True, True))), (If(v["arg4_ndim"] > 0, True, True)))) if n else
          And(And(And((If(v["arg1_ndim"] > 0, True, True)), (If(v["arg2_ndim"] > 0, True, True))), (If(v["arg3_ndim"] > 0, True, True))), (If(v["arg4_ndim"] > 0, True, True))))
)

def rule_100_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

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

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 100
        rule_100(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_100(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim']}, neg)
