import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if the input requires grad and it has more than 0 dimensions, it must be a floating point or complex dtype and should not be bool, or int8, int16, int32, int64, uint8  (Rule 73)

rule_73 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == True, And(And(And(And(And(And(And(v["arg1_ndim"] > 0, v["arg1_dtype"] != 0), v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), (Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))), True)) if n else
          If(v["arg2_value"] == True, And(And(And(And(And(And(And(v["arg1_ndim"] > 0, v["arg1_dtype"] != 0), v["arg1_dtype"] != 1), v["arg1_dtype"] != 2), v["arg1_dtype"] != 3), v["arg1_dtype"] != 4), v["arg1_dtype"] != 5), (Or(Or(Or(Or(v["arg1_dtype"] == 6, v["arg1_dtype"] == 7), v["arg1_dtype"] == 8), v["arg1_dtype"] == 9), v["arg1_dtype"] == 10))), True))
)

def rule_73_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 73
        rule_73(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_73(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
