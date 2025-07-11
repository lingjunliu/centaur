import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# value cannot overflow when converted to float16 (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(And(v["arg1_value"] > -6.55e+04, v["arg1_value"] < 6.55e+04)) if n else
          And(v["arg1_value"] > -6.55e+04, v["arg1_value"] < 6.55e+04))
)

def rule_12_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value']}, neg)
