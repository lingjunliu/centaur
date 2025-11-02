import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the datatype of first tensor is string the shape length of second tensor has to be one (Rule 140)

rule_140 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_dtype"] == 11, v["arg2_ndim"] == 1, True)) if n else
          If(v["arg1_dtype"] == 11, v["arg2_ndim"] == 1, True))
)

def rule_140_func(arg1, arg2, solver=None, neg=False):
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
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 140
        rule_140(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_140(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
