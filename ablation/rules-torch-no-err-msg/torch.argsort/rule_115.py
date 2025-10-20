import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If descending is false and stable is true, then dim must be specified (Rule 115)

rule_115 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == False, v["arg2_value"] == True), And((0 - v["arg4_ndim"]) <= v["arg3_value"], v["arg3_value"] < v["arg4_ndim"]), True)) if n else
          If(And(v["arg1_value"] == False, v["arg2_value"] == True), And((0 - v["arg4_ndim"]) <= v["arg3_value"], v["arg3_value"] < v["arg4_ndim"]), True))
)

def rule_115_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
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

        # Constraints for rule 115
        rule_115(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_ndim': arg4_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_115(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_ndim': arg4['ndim']}, neg)
