import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If `v_1` is a string "linear", then there exists an integer in the range [0, 10] such that `i` > 5 (Rule 51)

rule_51 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 20, Or([And(i < (10 + 1), i > 5) for i in range(6)]), True)) if n else
          If(v["arg1_value"] == 20, Or([And(i < (10 + 1), i > 5) for i in range(6)]), True))
)

def rule_51_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))

        # Constraints for rule 51
        rule_51(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_51(solver, {'arg1_value': arg1['value']}, neg)
