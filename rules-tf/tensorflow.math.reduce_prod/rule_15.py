import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# input_tensor should not be of bool type. Combined with axis range check and none axis check. (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_dtype"] != 0, (And(v["arg2_value"] >= (0 - v["arg1_ndim"]), Or(v["arg2_value"] < v["arg1_ndim"], If(v["arg2_value"] == -100, v["arg1_ndim"] > 0, True)))))) if n else
          And(v["arg1_dtype"] != 0, (And(v["arg2_value"] >= (0 - v["arg1_ndim"]), Or(v["arg2_value"] < v["arg1_ndim"], If(v["arg2_value"] == -100, v["arg1_ndim"] > 0, True))))))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 15
        rule_15(solver, {'arg1_dtype': arg1_dtype, 'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_dtype': arg1['dtype'], 'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
