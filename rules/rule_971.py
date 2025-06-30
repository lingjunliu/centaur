import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the reduction mode is not 'none' and the number of dimensions of input tensor is one, then the dtype must be in integer family (Rule 971)

rule_971 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_value"] != 6, v["arg1_ndim"] == 1), And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), False)) if n else
          If(And(v["arg2_value"] != 6, v["arg1_ndim"] == 1), And(1 <= v["arg1_dtype"], v["arg1_dtype"] <= 5), False))
)

def rule_971_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 971
        rule_971(solver, {'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_971(solver, {'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
