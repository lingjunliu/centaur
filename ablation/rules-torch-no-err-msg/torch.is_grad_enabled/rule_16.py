import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# is_grad_enabled returns a boolean, so its negation is also a boolean (Rule 16)

rule_16 = lambda s, v, n=False: (
    s.add(Not(Or([And(b < (1 + 1), If(v["arg1_value"], b == 1, b == 0)) for b in range(6)])) if n else
          Or([And(b < (1 + 1), If(v["arg1_value"], b == 1, b == 0)) for b in range(6)]))
)

def rule_16_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 16
        rule_16(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_16(solver, {'arg1_value': arg1['value']}, neg)
