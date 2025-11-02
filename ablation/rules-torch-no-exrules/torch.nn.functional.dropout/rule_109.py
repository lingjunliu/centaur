import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If p is 0 then training must be false relate to inplace boolean, ndim>0 (Rule 109)

rule_109 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0.0, And(And(v["arg2_value"] == False, v["arg3_value"] == False), v["arg4_ndim"] > 0), True)) if n else
          If(v["arg1_value"] == 0.0, And(And(v["arg2_value"] == False, v["arg3_value"] == False), v["arg4_ndim"] > 0), True))
)

def rule_109_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 109
        rule_109(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_109(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim']}, neg)
