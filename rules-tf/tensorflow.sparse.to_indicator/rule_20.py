import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# sp_input must be a tensor with int32 or int64 dtype, vocab_size must be a positive integer, and the max value of sp_input must be less than vocab_size, and minimum value of sp_input must be non-negative (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(v["arg1_ndim"] > 0, v["arg2_value"] > 0), v["arg2_value"] <= 9223372036854775807), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) < v["arg2_value"])) if n else
          And(And(And(And(And(v["arg1_ndim"] > 0, v["arg2_value"] > 0), v["arg2_value"] <= 9223372036854775807), (Or(v["arg1_dtype"] == 3, v["arg1_dtype"] == 4))), Select(v["arg1_range"], 0) >= 0), Select(v["arg1_range"], 1) < v["arg2_value"]))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
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
        arg1_range = Array('arg1_range', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        arg1_range = Store(arg1_range, 0, int(np.min(arg1)))
        arg1_range = Store(arg1_range, 1, int(np.max(arg1)))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 20
        rule_20(solver, {'arg1_range': arg1_range, 'arg1_ndim': arg1_ndim, 'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_range': arg1['range'], 'arg1_ndim': arg1['ndim'], 'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
