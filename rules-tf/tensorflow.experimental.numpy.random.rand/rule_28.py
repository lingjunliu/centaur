import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If shape is given as float or int, ensure it is greater than zero and less than some threshold (Rule 28)

rule_28 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_value"] > 0), (v["arg1_value"] < 100000))) if n else
          And((v["arg1_value"] > 0), (v["arg1_value"] < 100000)))
)

def rule_28_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 28
        rule_28(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_28(solver, {'arg1_value': arg1['value']}, neg)
