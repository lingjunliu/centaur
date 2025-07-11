import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If replace_global is true and input is a scalar, then pattern and rewrite must also be scalars (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] == True, v["arg1_ndim"] == 0), And(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0), False)) if n else
          If(And(v["arg4_value"] == True, v["arg1_ndim"] == 0), And(v["arg2_ndim"] == 0, v["arg3_ndim"] == 0), False))
)

def rule_39_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == arg4)

        # Constraints for rule 39
        rule_39(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
