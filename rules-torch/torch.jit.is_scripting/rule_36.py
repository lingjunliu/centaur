import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# if integer v_1 is 0 then every number is greater than 0 (Rule 36)

rule_36 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 0, And([Implies(x < (1 + 1), x > 0) for x in range(6)]), False)) if n else
          If(v["arg1_value"] == 0, And([Implies(x < (1 + 1), x > 0) for x in range(6)]), False))
)

def rule_36_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 36
        rule_36(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_36(solver, {'arg1_value': arg1['value']}, neg)
