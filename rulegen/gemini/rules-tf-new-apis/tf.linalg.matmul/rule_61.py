import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If a_is_sparse is True, the data type of a must be bfloat16 or float32 and rank must be 2 (Rule 61)

rule_61 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), v["arg1_ndim"] == 2), True)) if n else
          If(v["arg2_value"], And((Or(v["arg1_dtype"] == 7, v["arg1_dtype"] == 8)), v["arg1_ndim"] == 2), True))
)

def rule_61_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 61
        rule_61(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_61(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
