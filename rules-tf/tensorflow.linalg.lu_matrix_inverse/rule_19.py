import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If validate_args is true, and lower_upper is not complex, perm's dtype must be int32 or int64 and rank of perm should be N-1, where N is the rank of lower_upper. (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg3_value"] == True, v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), And(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), v["arg2_ndim"] == v["arg1_ndim"] - 1), False)) if n else
          If(And(And(v["arg3_value"] == True, v["arg1_dtype"] != 9), v["arg1_dtype"] != 10), And(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), v["arg2_ndim"] == v["arg1_ndim"] - 1), False))
)

def rule_19_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_value == arg3)

        # Constraints for rule 19
        rule_19(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
