import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the input, pattern, and rewrite are scalar tensors, then they each should have a dtype of string (Rule 66)

rule_66 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), And(And(v["arg1_dtype"] == 12, v["arg2_dtype"] == 12), v["arg3_dtype"] == 12), False)) if n else
          If(And(And(v["arg1_ndim"] == 0, v["arg2_ndim"] == 0), v["arg3_ndim"] == 0), And(And(v["arg1_dtype"] == 12, v["arg2_dtype"] == 12), v["arg3_dtype"] == 12), False))
)

def rule_66_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_ndim = Int('arg2_ndim')
        arg2_dtype = Int('arg2_dtype')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 66
        rule_66(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_dtype': arg2_dtype, 'arg2_ndim': arg2_ndim, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_66(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_dtype': arg2['dtype'], 'arg2_ndim': arg2['ndim'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim']}, neg)
