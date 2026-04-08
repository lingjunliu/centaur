import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Strides and dilations should have length N+2 and must be int list. (Rule 62)

rule_62 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_length"] == (v["arg3_ndim"] - v["arg4_value"] - 1) + 2, v["arg2_length"] == (v["arg3_ndim"] - v["arg4_value"] - 1) + 2)) if n else
          And(v["arg1_length"] == (v["arg3_ndim"] - v["arg4_value"] - 1) + 2, v["arg2_length"] == (v["arg3_ndim"] - v["arg4_value"] - 1) + 2))
)

def rule_62_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 62
        rule_62(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_62(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
