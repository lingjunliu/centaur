import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# if a float v_1 is greater than 0 and less than 1 and then all numbers are either zero or one (Rule 45)

rule_45 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] > 0, v["arg1_value"] < 1), And([Implies(x < (1 + 1), Or(x == 0, x == 1)) for x in range(6)]), False)) if n else
          If(And(v["arg1_value"] > 0, v["arg1_value"] < 1), And([Implies(x < (1 + 1), Or(x == 0, x == 1)) for x in range(6)]), False))
)

def rule_45_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 45
        rule_45(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_45(solver, {'arg1_value': arg1['value']}, neg)
