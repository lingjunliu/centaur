import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The input must be a valid tensor with a non-char dtype, and ndim > 0 when inplace is false or not specified.  Also check that inplace is a bool. (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_dtype"] != 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False))), If(v["arg2_value"], True, v["arg1_ndim"] > 0))) if n else
          And(And(v["arg1_dtype"] != 0, (Or(v["arg2_value"] == True, v["arg2_value"] == False))), If(v["arg2_value"], True, v["arg1_ndim"] > 0)))
)

def rule_37_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 37
        rule_37(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
