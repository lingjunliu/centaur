import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# torch.is_autocast_cpu_enabled: If the result of 'v_1 and true' is true, then v_1 must also be true. (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(If((And(v["arg1_value"], True)) == True, v["arg1_value"] == True, False)) if n else
          If((And(v["arg1_value"], True)) == True, v["arg1_value"] == True, False))
)

def rule_15_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 15
        rule_15(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_value': arg1['value']}, neg)
