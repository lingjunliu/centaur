import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The out_int32 parameter dictates the output tensor dtype and boundaries is 1D  (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(And(Or((And(v["arg2_value"] == True, v["arg1_dtype"] == 3)), (And(v["arg2_value"] == False, v["arg1_dtype"] == 4))), v["arg3_ndim"] == 1)) if n else
          And(Or((And(v["arg2_value"] == True, v["arg1_dtype"] == 3)), (And(v["arg2_value"] == False, v["arg1_dtype"] == 4))), v["arg3_ndim"] == 1))
)

def rule_44_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 44
        rule_44(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
