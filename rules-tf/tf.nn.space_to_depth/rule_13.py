import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# block_size and data_format are correlated with tensor dimensions (Rule 13)

rule_13 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(And(v["arg3_value"] == 33, v["arg1_ndim"] == 4), v["arg2_value"] >= 2)), (And(And(v["arg3_value"] == 34, v["arg1_ndim"] == 4), v["arg2_value"] >= 2))), (And(And(v["arg3_value"] == 37, v["arg1_ndim"] == 5), v["arg2_value"] >= 2)))) if n else
          Or(Or((And(And(v["arg3_value"] == 33, v["arg1_ndim"] == 4), v["arg2_value"] >= 2)), (And(And(v["arg3_value"] == 34, v["arg1_ndim"] == 4), v["arg2_value"] >= 2))), (And(And(v["arg3_value"] == 37, v["arg1_ndim"] == 5), v["arg2_value"] >= 2))))
)

def rule_13_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False
        if not isinstance(arg3, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Int('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == int(arg2))
        solver.add(arg3_value == list_of_string_values_tf.index(arg3))

        # Constraints for rule 13
        rule_13(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_13(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
