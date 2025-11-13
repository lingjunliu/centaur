import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check Features tensor's dtype and name parameters. (Rule 34)

rule_34 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(Or(Or(Or(Or(Or((v["arg1_dtype"] == 1), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 4)), (v["arg1_dtype"] == 5)), (v["arg1_dtype"] == 6)), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 13))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 11), v["arg2_value"] == 12), v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 15), v["arg2_value"] == 16), v["arg2_value"] == 17), v["arg2_value"] == 18), v["arg2_value"] == 19), v["arg2_value"] == 20), v["arg2_value"] == 21), v["arg2_value"] == 22), v["arg2_value"] == 23), v["arg2_value"] == 24), v["arg2_value"] == 25)))) if n else
          And((Or(Or(Or(Or(Or(Or(Or(Or((v["arg1_dtype"] == 1), (v["arg1_dtype"] == 2)), (v["arg1_dtype"] == 3)), (v["arg1_dtype"] == 4)), (v["arg1_dtype"] == 5)), (v["arg1_dtype"] == 6)), (v["arg1_dtype"] == 7)), (v["arg1_dtype"] == 8)), (v["arg1_dtype"] == 13))), (Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg2_value"] == 6, v["arg2_value"] == 11), v["arg2_value"] == 12), v["arg2_value"] == 13), v["arg2_value"] == 14), v["arg2_value"] == 15), v["arg2_value"] == 16), v["arg2_value"] == 17), v["arg2_value"] == 18), v["arg2_value"] == 19), v["arg2_value"] == 20), v["arg2_value"] == 21), v["arg2_value"] == 22), v["arg2_value"] == 23), v["arg2_value"] == 24), v["arg2_value"] == 25))))
)

def rule_34_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 34
        rule_34(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_34(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value']}, neg)
