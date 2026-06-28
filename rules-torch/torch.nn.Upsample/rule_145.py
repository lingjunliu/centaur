import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Check input dimension is at least 3D, if the align_corners is enabled along the mode (Rule 145)

rule_145 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == True, And(v["arg2_ndim"] >= 3, (Or(Or(Or(v["arg3_value"] == 20, v["arg3_value"] == 26), v["arg3_value"] == 27), v["arg3_value"] == 28))), True)) if n else
          If(v["arg1_value"] == True, And(v["arg2_ndim"] >= 3, (Or(Or(Or(v["arg3_value"] == 20, v["arg3_value"] == 26), v["arg3_value"] == 27), v["arg3_value"] == 28))), True))
)

def rule_145_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_string_values_torch.index(arg3))

        # Constraints for rule 145
        rule_145(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_145(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
