import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# indices is int64 and 2D, shape, start, size are int64 and 1D, values are 1D (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(v["arg1_dtype"] == 4, v["arg1_ndim"] == 2), v["arg2_dtype"] == 4), v["arg2_ndim"] == 1), v["arg3_dtype"] == 4), v["arg3_ndim"] == 1), v["arg4_dtype"] == 4), v["arg4_ndim"] == 1), v["arg5_ndim"] == 1)) if n else
          And(And(And(And(And(And(And(And(v["arg1_dtype"] == 4, v["arg1_ndim"] == 2), v["arg2_dtype"] == 4), v["arg2_ndim"] == 1), v["arg3_dtype"] == 4), v["arg3_ndim"] == 1), v["arg4_dtype"] == 4), v["arg4_ndim"] == 1), v["arg5_ndim"] == 1))
)

def rule_61_func(arg1, arg2, arg3, arg4, arg5, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_dtype = Int('arg4_dtype')
        arg5_ndim = Int('arg5_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))
        solver.add(arg5_ndim == arg5.ndim)

        # Constraints for rule 61
        rule_61(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype, 'arg5_ndim': arg5_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype'], 'arg5_ndim': arg5['ndim']}, neg)
