import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if inplace is true and if dtype is a complex number, then its dimensions must be less than or equal to 2 (Rule 99)

rule_99 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), v["arg2_value"]), v["arg1_ndim"] <= 2, True)) if n else
          If(And((Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10)), v["arg2_value"]), v["arg1_ndim"] <= 2, True))
)

def rule_99_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)

        # Constraints for rule 99
        rule_99(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_99(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
