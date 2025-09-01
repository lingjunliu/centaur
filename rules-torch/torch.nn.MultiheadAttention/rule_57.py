import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If is_causal is true, attn_mask must be defined and have compatible dimensions with query (Rule 57)

rule_57 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(v["arg3_ndim"] > 0, (If(v["arg2_ndim"] == 3, (Or(v["arg3_ndim"] == 2, v["arg3_ndim"] == 3)), v["arg3_ndim"] == 2))), True)) if n else
          If(v["arg1_value"] == True, And(v["arg3_ndim"] > 0, (If(v["arg2_ndim"] == 3, (Or(v["arg3_ndim"] == 2, v["arg3_ndim"] == 3)), v["arg3_ndim"] == 2))), True))
)

def rule_57_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_ndim = Int('arg3_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_ndim == arg3.ndim)

        # Constraints for rule 57
        rule_57(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_ndim': arg3_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_57(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_ndim': arg3['ndim']}, neg)
