import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If output tensor is provided and base is a float or complex type, it should exist (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), v["arg2_ndim"] > 0, True)) if n else
          If(And(6 <= v["arg1_dtype"], v["arg1_dtype"] <= 10), v["arg2_ndim"] > 0, True))
)

def rule_111_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 111
        rule_111(solver, {'arg1_dtype': arg1_dtype, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_dtype': arg1['dtype'], 'arg2_ndim': arg2['ndim']}, neg)
