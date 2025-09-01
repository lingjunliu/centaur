import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If pad_to_bucket_boundary is true and drop_remainder is true, the padding values should be a scalar tensor with a valid type. (Rule 82)

rule_82 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"], v["arg2_value"]), And(v["arg3_ndim"] == 0, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), v["arg3_dtype"] == 6), v["arg3_dtype"] == 7), v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10), v["arg3_dtype"] == 11))), True)) if n else
          If(And(v["arg1_value"], v["arg2_value"]), And(v["arg3_ndim"] == 0, (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg3_dtype"] == 1, v["arg3_dtype"] == 2), v["arg3_dtype"] == 3), v["arg3_dtype"] == 4), v["arg3_dtype"] == 5), v["arg3_dtype"] == 6), v["arg3_dtype"] == 7), v["arg3_dtype"] == 8), v["arg3_dtype"] == 9), v["arg3_dtype"] == 10), v["arg3_dtype"] == 11))), True))
)

def rule_82_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_ndim == arg3.ndim)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 82
        rule_82(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_82(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_dtype': arg3['dtype']}, neg)
