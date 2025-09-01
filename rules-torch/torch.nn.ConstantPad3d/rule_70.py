import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Padding as tuple: if total padding is large, then each element of the tuple cannot be too big, avoid redundant vars. (Rule 70)

rule_70 = lambda s, v, n=False: (
    s.add(Not(If((Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1) + Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3) + Select(v["arg1_values"], 4) + Select(v["arg1_values"], 5)) > 1000, And([Implies(i < (5 + 1), Select(v["arg1_values"], i) < 1000) for i in range(6)]), True)) if n else
          If((Select(v["arg1_values"], 0) + Select(v["arg1_values"], 1) + Select(v["arg1_values"], 2) + Select(v["arg1_values"], 3) + Select(v["arg1_values"], 4) + Select(v["arg1_values"], 5)) > 1000, And([Implies(i < (5 + 1), Select(v["arg1_values"], i) < 1000) for i in range(6)]), True))
)

def rule_70_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 70
        rule_70(solver, {'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_70(solver, {'arg1_values': arg1['values']}, neg)
