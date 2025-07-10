import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If str is valid matrix multiplication, then the rank of input tensors should be 2 or the rank of input tensors should not be 2 (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If((Or(v["arg3_value"] == 3, v["arg3_value"] == 5)), (And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2)), (And(v["arg1_ndim"] != 2, v["arg2_ndim"] != 2)))) if n else
          If((Or(v["arg3_value"] == 3, v["arg3_value"] == 5)), (And(v["arg1_ndim"] == 2, v["arg2_ndim"] == 2)), (And(v["arg1_ndim"] != 2, v["arg2_ndim"] != 2))))
)

def rule_45_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = String('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == list_of_string_values.index(arg3))

        # Constraints for rule 45
        rule_45(solver, {'arg1_ndim': arg1_ndim, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_ndim': arg1['ndim'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
