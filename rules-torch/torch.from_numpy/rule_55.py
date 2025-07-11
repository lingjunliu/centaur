import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# Example with string, not equal to none or mean or max or sum (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(And(And(And(v["arg1_value"] != 6, v["arg1_value"] != 7), v["arg1_value"] != 9), v["arg1_value"] != 8)) if n else
          And(And(And(v["arg1_value"] != 6, v["arg1_value"] != 7), v["arg1_value"] != 9), v["arg1_value"] != 8))
)

def rule_55_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.torch.index(arg1))

        # Constraints for rule 55
        rule_55(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_value': arg1['value']}, neg)
