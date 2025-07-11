import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If output type is specified as tf.int32 or tf.int64, the returned tensor has to be a 1D tensor (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), v["arg2_ndim"] == 1, False)) if n else
          If(Or(v["arg1_value"] == 3, v["arg1_value"] == 4), v["arg2_ndim"] == 1, False))
)

def rule_12_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_ndim = Int('arg2_ndim')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_ndim == arg2.ndim)

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim']}, neg)
