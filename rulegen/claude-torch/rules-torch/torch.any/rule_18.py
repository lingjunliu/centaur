import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# output tensor dimension matches input when keepdim is true (Rule 18)

rule_18 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg2_value"] == False, v["arg3_ndim"] == v["arg1_ndim"])) if n else
          Or(v["arg2_value"] == False, v["arg3_ndim"] == v["arg1_ndim"]))
)

def rule_18_func(arg1, arg2, arg3, solver=None, neg=False):
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
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 18
        rule_18(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_18(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim']}, neg)
