import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# SymInt check: prevent large negative value to prevent overflow, changed comparision method and constant and added more checks if value inside range or outside, using and operator. (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] < -5425825916322441126) if n else
          v["arg1_value"] < -5425825916322441126)
)

def rule_72_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value']}, neg)
