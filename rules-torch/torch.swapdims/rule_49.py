import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The dimensions must be valid and different (Rule 49)

rule_49 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(0 - v["arg1_ndim"] <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]), 0 - v["arg1_ndim"] <= v["arg3_value"]), v["arg3_value"] < v["arg1_ndim"]), v["arg2_value"] != v["arg3_value"])) if n else
          And(And(And(And(0 - v["arg1_ndim"] <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]), 0 - v["arg1_ndim"] <= v["arg3_value"]), v["arg3_value"] < v["arg1_ndim"]), v["arg2_value"] != v["arg3_value"]))
)

def rule_49_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 49
        rule_49(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_49(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
