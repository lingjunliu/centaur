import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If input and out are defined, and dim is a tuple, then if keepdim is false, the output's dimensions is the input's dimensions minus the length of the dim tuple (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(If(v["arg3_value"] == False, v["arg4_ndim"] == v["arg1_ndim"] - v["arg2_length"], True)) if n else
          If(v["arg3_value"] == False, v["arg4_ndim"] == v["arg1_ndim"] - v["arg2_length"], True))
)

def rule_49_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_value = Bool('arg3_value')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 49
        rule_49(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim']}, neg)
