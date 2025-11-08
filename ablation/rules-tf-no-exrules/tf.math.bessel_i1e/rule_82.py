import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If x is a tensor and of dtype float16, the max should be <= 100, if x is a tensor and of dtype float32 the min should be >= -100, If x is a tensor and dtype is float64 ndim > 0. if x is a tensor but none of this applies then its valid. (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or((And(v["arg1_dtype"] == 7, Select(v["arg1_range"], 1) <= 100)), (And(v["arg1_dtype"] == 8, Select(v["arg1_range"], 0) >= -100))), (And(v["arg1_dtype"] == 9, v["arg1_ndim"] > 0))), (And(And(v["arg1_dtype"] != 7, v["arg1_dtype"] != 8), v["arg1_dtype"] != 9)))) if n else
          Or(Or(Or((And(v["arg1_dtype"] == 7, Select(v["arg1_range"], 1) <= 100)), (And(v["arg1_dtype"] == 8, Select(v["arg1_range"], 0) >= -100))), (And(v["arg1_dtype"] == 9, v["arg1_ndim"] > 0))), (And(And(v["arg1_dtype"] != 7, v["arg1_dtype"] != 8), v["arg1_dtype"] != 9))))
)

def rule_82_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg1_range = Array('arg1_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))

        # Constraints for rule 82
        rule_82(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg1_range': arg1_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg1_range': arg1['range']}, neg)
