import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If out is provided, when keepdim is False, all dimensions in v3 are reduced then the number of v3's dimensions == original dimensions-dim (Rule 64)

rule_64 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == False, v["arg2_ndim"] == v["arg1_ndim"] - v["arg3_length"], True)) if n else
          If(v["arg4_value"] == False, v["arg2_ndim"] == v["arg1_ndim"] - v["arg3_length"], True))
)

def rule_64_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_length = Int('arg3_length')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_length == len(arg3))
        solver.add(arg4_value == arg4)

        # Constraints for rule 64
        rule_64(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_64(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length'], 'arg4_value': arg4['value']}, neg)
