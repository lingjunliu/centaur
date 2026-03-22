import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Use a quantified expression to guarantee that the indices tensor has the appropriate type (int32 or int64 (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (v["arg1_ndim"] - 1 + 1), Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 4)) for i in range(6)])) if n else
          Or([And(i < (v["arg1_ndim"] - 1 + 1), Or(v["arg1_dtype"] == 2, v["arg1_dtype"] == 4)) for i in range(6)]))
)

def rule_62_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))

        # Constraints for rule 62
        rule_62(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim']}, neg)
