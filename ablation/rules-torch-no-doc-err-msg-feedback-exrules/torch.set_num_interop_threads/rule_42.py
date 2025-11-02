import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If using Intel MKL, interop threads should be power of 2 for optimal performance. mkl_enabled is bool (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], Or([And(i < (8 + 1), v["arg1_value"] == (2 * i)) for i in range(6)]), True)) if n else
          If(v["arg2_value"], Or([And(i < (8 + 1), v["arg1_value"] == (2 * i)) for i in range(6)]), True))
)

def rule_42_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Bool('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
