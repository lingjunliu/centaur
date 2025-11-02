import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If dim is a tuple then keepdim should be boolean, prevents incorrect arg combinations (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg2_value"] == True, v["arg2_value"] == False)) if n else
          Or(v["arg2_value"] == True, v["arg2_value"] == False))
)

def rule_55_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg2_value == arg2)

        # Constraints for rule 55
        rule_55(solver, {'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg2_value': arg2['value']}, neg)
