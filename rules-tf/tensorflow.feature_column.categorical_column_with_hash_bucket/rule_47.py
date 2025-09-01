import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# hash_bucket_size and dtype combination check (Rule 47)

rule_47 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg1_value"] % 1 == 0), (v["arg1_value"] > 1)), (Or(Or(Or(Or(Or((v["arg2_value"] == 1), (v["arg2_value"] == 2)), (v["arg2_value"] == 3)), (v["arg2_value"] == 4)), (v["arg2_value"] == 5)), (v["arg2_value"] == 11))))) if n else
          And(And((v["arg1_value"] % 1 == 0), (v["arg1_value"] > 1)), (Or(Or(Or(Or(Or((v["arg2_value"] == 1), (v["arg2_value"] == 2)), (v["arg2_value"] == 3)), (v["arg2_value"] == 4)), (v["arg2_value"] == 5)), (v["arg2_value"] == 11)))))
)

def rule_47_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not ((isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)) or isinstance(arg1, (float, np.floating))):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 47
        rule_47(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_47(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
