import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# max_norm must be bigger than 0 and finite (Rule 104)

rule_104 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] > 0, v["arg1_value"] < 3.402823e+38)) if n else
          And(v["arg1_value"] > 0, v["arg1_value"] < 3.402823e+38))
)

def rule_104_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (float, np.floating)) or (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool))):
            return False

        # Variable declarations
        solver = Solver()

        # Value assignments

        # Constraints for rule 104
        rule_104(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_104(solver, {'arg1_value': arg1['value']}, neg)
