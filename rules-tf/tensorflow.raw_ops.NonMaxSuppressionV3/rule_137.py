import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# boxes and scores be same dtype half or float32, iou_threshold, score_threshold are scalar (Rule 137)

rule_137 = lambda s, v, n=False: (
    s.add(Not(And(And(And((Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg3_ndim"] == 0), v["arg4_ndim"] == 0)) if n else
          And(And(And((Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7)), v["arg1_dtype"] == v["arg2_dtype"]), v["arg3_ndim"] == 0), v["arg4_ndim"] == 0))
)

def rule_137_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 137
        rule_137(solver, {'arg1_dtype': arg1_dtype, 'arg2_dtype': arg2_dtype, 'arg3_ndim': arg3_ndim, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_137(solver, {'arg1_dtype': arg1['dtype'], 'arg2_dtype': arg2['dtype'], 'arg3_ndim': arg3['ndim'], 'arg4_ndim': arg4['ndim']}, neg)
