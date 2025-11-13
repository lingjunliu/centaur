import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if axis is specified then it should be an integer and not any string value from the specific list and is applicable only if dimension of the tensor is >0 (Rule 84)

rule_84 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] > 0, (And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 21, v["arg2_value"] != 22), v["arg2_value"] != 23), v["arg2_value"] != 24), v["arg2_value"] != 25), v["arg2_value"] != 6), v["arg2_value"] != 7), v["arg2_value"] != 8), v["arg2_value"] != 9), v["arg2_value"] != 10), v["arg2_value"] != 11), v["arg2_value"] != 12), v["arg2_value"] != 13), v["arg2_value"] != 14), v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 17), v["arg2_value"] != 18), v["arg2_value"] != 19), v["arg2_value"] != 20)), True)) if n else
          If(v["arg1_ndim"] > 0, (And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg2_value"] != 21, v["arg2_value"] != 22), v["arg2_value"] != 23), v["arg2_value"] != 24), v["arg2_value"] != 25), v["arg2_value"] != 6), v["arg2_value"] != 7), v["arg2_value"] != 8), v["arg2_value"] != 9), v["arg2_value"] != 10), v["arg2_value"] != 11), v["arg2_value"] != 12), v["arg2_value"] != 13), v["arg2_value"] != 14), v["arg2_value"] != 15), v["arg2_value"] != 16), v["arg2_value"] != 17), v["arg2_value"] != 18), v["arg2_value"] != 19), v["arg2_value"] != 20)), True))
)

def rule_84_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, str) or (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool))):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)

        # Constraints for rule 84
        rule_84(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_84(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
