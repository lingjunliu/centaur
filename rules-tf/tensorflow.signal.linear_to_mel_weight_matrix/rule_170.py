import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the sampling rate is low, and only if it is. then you should prevent high Hz to cause problems on calculations. (Rule 170)

rule_170 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] < 44100, v["arg1_value"] < 20000, True)) if n else
          If(v["arg1_value"] < 44100, v["arg1_value"] < 20000, True))
)

def rule_170_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 170
        rule_170(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_170(solver, {'arg1_value': arg1['value']}, neg)
