import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if axis is specified then it should be an integer and not any string value from the specific list. (Rule 76)

rule_76 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 21, v["arg1_value"] == 22), v["arg1_value"] == 23), v["arg1_value"] == 24), v["arg1_value"] == 25), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11), v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), v["arg1_value"] == 20), False, True)) if n else
          If(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(Or(v["arg1_value"] == 21, v["arg1_value"] == 22), v["arg1_value"] == 23), v["arg1_value"] == 24), v["arg1_value"] == 25), v["arg1_value"] == 6), v["arg1_value"] == 7), v["arg1_value"] == 8), v["arg1_value"] == 9), v["arg1_value"] == 10), v["arg1_value"] == 11), v["arg1_value"] == 12), v["arg1_value"] == 13), v["arg1_value"] == 14), v["arg1_value"] == 15), v["arg1_value"] == 16), v["arg1_value"] == 17), v["arg1_value"] == 18), v["arg1_value"] == 19), v["arg1_value"] == 20), False, True))
)

def rule_76_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, str) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 76
        rule_76(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_76(solver, {'arg1_value': arg1['value']}, neg)
