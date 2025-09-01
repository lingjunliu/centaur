import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If sample_rate is a multiple of 1000, then it also should be divisible by 44100 for synchronization with other audio devices (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] % 1000 == 0, v["arg1_value"] % 44100 == 0, True)) if n else
          If(v["arg1_value"] % 1000 == 0, v["arg1_value"] % 44100 == 0, True))
)

def rule_33_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 33
        rule_33(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_value': arg1['value']}, neg)
