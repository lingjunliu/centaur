import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If images is a 4-D tensor and boxes is a 3-D tensor, then if boxes is of type float32, then the minimum value of boxes should be less than or equal to the maximum value of boxes (Rule 118)

rule_118 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 3)), If((v["arg2_dtype"] == 7), Select(v["arg2_range"], 0) <= Select(v["arg2_range"], 1), True), True)) if n else
          If((And(v["arg1_ndim"] == 4, v["arg2_ndim"] == 3)), If((v["arg2_dtype"] == 7), Select(v["arg2_range"], 0) <= Select(v["arg2_range"], 1), True), True))
)

def rule_118_func(arg1, arg2, solver=None, neg=False):
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
        arg2_dtype = Int('arg2_dtype')
        arg2_range = Array('arg2_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        arg2_range = Store(arg2_range, 0, int(np.min(arg2)))
        arg2_range = Store(arg2_range, 1, int(np.max(arg2)))

        # Constraints for rule 118
        rule_118(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg2_dtype': arg2_dtype, 'arg2_range': arg2_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_118(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_range': arg2['range']}, neg)
