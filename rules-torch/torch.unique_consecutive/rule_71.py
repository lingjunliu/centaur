import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Combining dimension restrictions and dtype restrictions (Rule 71)

rule_71 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg4_value"] >= (0 - v["arg1_ndim"]), v["arg4_value"] < v["arg1_ndim"]), And(And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), If(Or(v["arg2_value"] == True, v["arg3_value"] == True), v["arg1_ndim"] > 0, True)), True)) if n else
          If(And(v["arg4_value"] >= (0 - v["arg1_ndim"]), v["arg4_value"] < v["arg1_ndim"]), And(And(v["arg1_dtype"] != 9, v["arg1_dtype"] != 10), If(Or(v["arg2_value"] == True, v["arg3_value"] == True), v["arg1_ndim"] > 0, True)), True))
)

def rule_71_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 71
        rule_71(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_71(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
