import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Handle type error and combination of arguments, final rule (Rule 54)

rule_54 = lambda s, v, n=False: (
    s.add(Not(If(And((0 - v["arg1_ndim"]) <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]), If(v["arg4_value"] == True, And(v["arg3_dtype"] == 5, v["arg1_ndim"] == v["arg3_ndim"]), v["arg1_ndim"] == v["arg3_ndim"]), False)) if n else
          If(And((0 - v["arg1_ndim"]) <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]), If(v["arg4_value"] == True, And(v["arg3_dtype"] == 5, v["arg1_ndim"] == v["arg3_ndim"]), v["arg1_ndim"] == v["arg3_ndim"]), False))
)

def rule_54_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, np.ndarray):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))
        solver.add(arg4_value == arg4)

        # Constraints for rule 54
        rule_54(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype, 'arg3_ndim': arg3_ndim, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_54(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype'], 'arg3_ndim': arg3['ndim'], 'arg4_value': arg4['value']}, neg)
