import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If p is 0 then training must be false relate to inplace boolean, ndim>0, and require float tensor (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0.0, And(And(And(v["arg2_value"] == False, v["arg3_value"] == False), v["arg4_ndim"] > 0), (Or(Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8), v["arg4_dtype"] == 6))), True)) if n else
          If(v["arg1_value"] == 0.0, And(And(And(v["arg2_value"] == False, v["arg3_value"] == False), v["arg4_ndim"] > 0), (Or(Or(v["arg4_dtype"] == 7, v["arg4_dtype"] == 8), v["arg4_dtype"] == 6))), True))
)

def rule_115_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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
        arg4_dtype = Int('arg4_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_ndim == arg4.ndim)
        solver.add(arg4_dtype == list_of_available_dtypes.index(arg4.dtype))

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_dtype': arg4_dtype, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_dtype': arg4['dtype'], 'arg4_ndim': arg4['ndim']}, neg)
