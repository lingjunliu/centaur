import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Either autocast_cpu_dtype is specified or high_precision is specified, or neither, but not both (Rule 37)

rule_37 = lambda s, v, n=False: (
    s.add(Not(Or((And(v["arg1_value"] == -1, (Or(v["arg2_value"] == False, v["arg2_value"] == True)))), (And(v["arg1_value"] != -1, v["arg2_value"] == False)))) if n else
          Or((And(v["arg1_value"] == -1, (Or(v["arg2_value"] == False, v["arg2_value"] == True)))), (And(v["arg1_value"] != -1, v["arg2_value"] == False))))
)

def rule_37_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 37
        rule_37(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_37(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
