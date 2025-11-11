import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# for all valid month indices, day must lie within 1–31 (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (12 + 1), Or((v["arg1_value"] != i), (And(1 <= v["arg2_value"], v["arg2_value"] <= 31)))) for i in range(6)])) if n else
          And([Implies(i < (12 + 1), Or((v["arg1_value"] != i), (And(1 <= v["arg2_value"], v["arg2_value"] <= 31)))) for i in range(6)]))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 15
        rule_15(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
