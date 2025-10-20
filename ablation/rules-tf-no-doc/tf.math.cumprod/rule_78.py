import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If reverse is true, exclusive cannot also be true when axis equals negative one (Rule 78)

rule_78 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, v["arg2_value"] == True), v["arg3_value"] != 0 - v["arg4_ndim"], True)) if n else
          If(And(v["arg1_value"] == True, v["arg2_value"] == True), v["arg3_value"] != 0 - v["arg4_ndim"], True))
)

def rule_78_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not isinstance(arg4, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')
        arg4_ndim = Int('arg4_ndim')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_ndim == arg4.ndim)

        # Constraints for rule 78
        rule_78(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_78(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim']}, neg)
