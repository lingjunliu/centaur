import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# alpha, l1 and l2 are scalars with compatible dtypes with the var tensor (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"])) if n else
          And(And(And(And(And(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0), v["arg4_ndim"] == 0), v["arg1_dtype"] == v["arg2_dtype"]), v["arg1_dtype"] == v["arg3_dtype"]), v["arg1_dtype"] == v["arg4_dtype"]))
)

def rule_16_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg4_ndim = Int('arg4_ndim')
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 16
        rule_16(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype, 'arg4_ndim': arg4_ndim, 'arg4_dtype': arg4_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype'], 'arg4_ndim': arg4['ndim'], 'arg4_dtype': arg4['dtype']}, neg)
