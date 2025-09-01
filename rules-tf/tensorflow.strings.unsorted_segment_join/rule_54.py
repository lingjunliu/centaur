import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if segment_ids is a scalar, inputs must be a vector to avoid dimension mismatch and segment_ids has smaller dimension than inputs, if not scalar, and dtype must be string (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_dtype"] == 12), (If(v["arg2_ndim"] == 0, v["arg1_ndim"] == 1, v["arg2_ndim"] <= v["arg1_ndim"])))) if n else
          And((v["arg1_dtype"] == 12), (If(v["arg2_ndim"] == 0, v["arg1_ndim"] == 1, v["arg2_ndim"] <= v["arg1_ndim"]))))
)

def rule_54_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 54
        rule_54(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
