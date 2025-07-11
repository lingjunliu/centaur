import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If keepdim is True then the dimension should be a valid dimension, and the dim needs to be an integer (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg3_value"] == True, v["arg2_value"] > (v["arg2_value"] - 1)), v["arg2_value"] < (v["arg2_value"] + 1)), And((0 - v["arg1_ndim"]) <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]), False)) if n else
          If(And(And(v["arg3_value"] == True, v["arg2_value"] > (v["arg2_value"] - 1)), v["arg2_value"] < (v["arg2_value"] + 1)), And((0 - v["arg1_ndim"]) <= v["arg2_value"], v["arg2_value"] < v["arg1_ndim"]), False))
)

def rule_33_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not ((isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)) or isinstance(arg2, (float, np.floating))):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg3_value == arg3)

        # Constraints for rule 33
        rule_33(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
