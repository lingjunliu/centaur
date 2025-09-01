import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if shape is a scalar, means, stdevs, minvals, maxvals must also be scalars of corresponding type (Rule 17)

rule_17 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 0, (And(And(And(And(And(And(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), v["arg5_ndim"] == 0), v["arg2_dtype"] == v["arg3_dtype"]), v["arg2_dtype"] == v["arg4_dtype"]), v["arg2_dtype"] == v["arg5_dtype"])), True)) if n else
          If(v["arg1_ndim"] == 0, (And(And(And(And(And(And(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), v["arg5_ndim"] == 0), v["arg2_dtype"] == v["arg3_dtype"]), v["arg2_dtype"] == v["arg4_dtype"]), v["arg2_dtype"] == v["arg5_dtype"])), True))
)

def rule_17_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_dtype = Int('arg4_dtype')
        arg5_ndim = Int('arg5_ndim')
        arg5_dtype = Int('arg5_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_ndim == arg5.ndim)
        solver.add(arg5_dtype == list_of_available_dtypes.index(arg5.dtype))

        # Constraints for rule 17
        rule_17(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype, 'arg5_ndim': arg5_ndim, 'arg5_dtype': arg5_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_17(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype'], 'arg5_ndim': arg5['ndim'], 'arg5_dtype': arg5['dtype']}, neg)
