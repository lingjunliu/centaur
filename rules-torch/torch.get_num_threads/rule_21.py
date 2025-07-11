import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# torch.get_num_threads accepts no parameters, therefore any bool v_1 passed to it must either be true or false, at least one of the two cases hold. (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(Or(v["arg1_value"], (v["arg1_value"] == False))) if n else
          Or(v["arg1_value"], (v["arg1_value"] == False)))
)

def rule_21_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')

        # Value assignments
        solver.add(arg1_value == arg1)

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value']}, neg)
