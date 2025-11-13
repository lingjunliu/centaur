import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Has to be a valid Spectrogram bins, power 2 values which can be checked through log (Rule 144)

rule_144 = lambda s, v, n=False: (
    s.add(Not(Or([And(i < (15 + 1), v["arg1_value"] == 2 * i) for i in range(6)])) if n else
          Or([And(i < (15 + 1), v["arg1_value"] == 2 * i) for i in range(6)]))
)

def rule_144_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 144
        rule_144(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_144(solver, {'arg1_value': arg1['value']}, neg)
