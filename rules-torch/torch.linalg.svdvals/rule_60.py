import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the output dimension does not follow certain rules raise error (Rule 60)

rule_60 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg3_value"] == v["arg1_ndim"], v["arg4_value"] == v["arg2_ndim"]), If(v["arg3_value"] > 2, v["arg4_value"] == v["arg3_value"] - 1, If(v["arg3_value"] == 2, v["arg4_value"] == 1, False)))) if n else
          And(And(v["arg3_value"] == v["arg1_ndim"], v["arg4_value"] == v["arg2_ndim"]), If(v["arg3_value"] > 2, v["arg4_value"] == v["arg3_value"] - 1, If(v["arg3_value"] == 2, v["arg4_value"] == 1, False))))
)

def rule_60_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 60
        rule_60(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_60(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
