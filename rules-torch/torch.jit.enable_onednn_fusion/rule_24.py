import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# enabled parameter should be one of two values, where one value causes fusion and the other does not (Rule 24)

rule_24 = lambda s, v, n=False: (
    s.add(Not(Or([And(a < (1 + 1), And((If(a == 0, v["arg1_value"] == True, v["arg1_value"] == False)), If(v["arg1_value"] == True, Or([And(b < (1 + 1), b == 1) for b in range(6)]), Or([And(c < (1 + 1), c == 0) for c in range(6)])))) for a in range(6)])) if n else
          Or([And(a < (1 + 1), And((If(a == 0, v["arg1_value"] == True, v["arg1_value"] == False)), If(v["arg1_value"] == True, Or([And(b < (1 + 1), b == 1) for b in range(6)]), Or([And(c < (1 + 1), c == 0) for c in range(6)])))) for a in range(6)]))
)

def rule_24_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 24
        rule_24(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_24(solver, {'arg1_value': arg1['value']}, neg)
