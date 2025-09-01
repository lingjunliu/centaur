import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# dtype must be one of the supported types and relate to the perm tensor's dtype if provided (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 12)), (If(v["arg1_value"] != 12, v["arg2_dtype"] == v["arg1_value"], True)))) if n else
          And((Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 1, v["arg1_value"] == 2), v["arg1_value"] == 3), v["arg1_value"] == 4), v["arg1_value"] == 5), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 12)), (If(v["arg1_value"] != 12, v["arg2_dtype"] == v["arg1_value"], True))))
)

def rule_19_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value'], 'arg2_dtype': arg2['dtype']}, neg)
