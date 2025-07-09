import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Length + Relation needs to relate with Tensor + checks of validity of this certain string parameter - so that value doesn't exceed. (Rule 124)

rule_124 = lambda s, v, n=False: (
    s.add(Not(If(v["arg4_value"] == 11, If(v["arg3_ndim"] > 0, v["arg1_value"] > v["arg2_length"], v["arg1_value"] > 0), False)) if n else
          If(v["arg4_value"] == 11, If(v["arg3_ndim"] > 0, v["arg1_value"] > v["arg2_length"], v["arg1_value"] > 0), False))
)

def rule_124_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_ndim = Int('arg3_ndim')
        arg4_value = String('arg4_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg4_value == list_of_string_values.index(arg4))

        # Constraints for rule 124
        rule_124(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_124(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
