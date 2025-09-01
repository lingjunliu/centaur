import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The size of the hash table used for cache keys is a prime number greater than 100 (Rule 91)

rule_91 = lambda s, v, n=False: (
    s.add(Not((Or(Or(Or(Or(v["arg1_value"] == 101, v["arg1_value"] == 103), v["arg1_value"] == 107), v["arg1_value"] == 109), v["arg1_value"] == 113))) if n else
          (Or(Or(Or(Or(v["arg1_value"] == 101, v["arg1_value"] == 103), v["arg1_value"] == 107), v["arg1_value"] == 109), v["arg1_value"] == 113)))
)

def rule_91_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 91
        rule_91(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_91(solver, {'arg1_value': arg1['value']}, neg)
