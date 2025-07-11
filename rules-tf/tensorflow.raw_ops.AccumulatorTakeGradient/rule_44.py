import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# num_required must be a prime number or 1 (Rule 44)

rule_44 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((v["arg1_value"] == 1), (v["arg1_value"] == 2)), (v["arg1_value"] == 3)), (v["arg1_value"] == 5)), (v["arg1_value"] == 7)), (v["arg1_value"] == 11)), (v["arg1_value"] == 13)), (v["arg1_value"] == 17)), (v["arg1_value"] == 19)), (v["arg1_value"] == 23)), (v["arg1_value"] == 29)), (v["arg1_value"] == 31)), (v["arg1_value"] == 37)), (v["arg1_value"] == 41)), (v["arg1_value"] == 43)), (v["arg1_value"] == 47)), (v["arg1_value"] == 53)), (v["arg1_value"] == 59)), (v["arg1_value"] == 61)), (v["arg1_value"] == 67)), (v["arg1_value"] == 71)), (v["arg1_value"] == 73)), (v["arg1_value"] == 79)), (v["arg1_value"] == 83)), (v["arg1_value"] == 89)), (v["arg1_value"] == 97))) if n else
          Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or((v["arg1_value"] == 1), (v["arg1_value"] == 2)), (v["arg1_value"] == 3)), (v["arg1_value"] == 5)), (v["arg1_value"] == 7)), (v["arg1_value"] == 11)), (v["arg1_value"] == 13)), (v["arg1_value"] == 17)), (v["arg1_value"] == 19)), (v["arg1_value"] == 23)), (v["arg1_value"] == 29)), (v["arg1_value"] == 31)), (v["arg1_value"] == 37)), (v["arg1_value"] == 41)), (v["arg1_value"] == 43)), (v["arg1_value"] == 47)), (v["arg1_value"] == 53)), (v["arg1_value"] == 59)), (v["arg1_value"] == 61)), (v["arg1_value"] == 67)), (v["arg1_value"] == 71)), (v["arg1_value"] == 73)), (v["arg1_value"] == 79)), (v["arg1_value"] == 83)), (v["arg1_value"] == 89)), (v["arg1_value"] == 97)))
)

def rule_44_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 44
        rule_44(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_44(solver, {'arg1_value': arg1['value']}, neg)
