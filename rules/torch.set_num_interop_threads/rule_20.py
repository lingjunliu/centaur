import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If the number of interop threads is v_1, it should be a power of 2 if v_2 is "tanh" (Rule 20)

rule_20 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 11, Or([And(i < (10 + 1), v["arg1_value"] == 2 * i) for i in range(6)]), False)) if n else
          If(v["arg2_value"] == 11, Or([And(i < (10 + 1), v["arg1_value"] == 2 * i) for i in range(6)]), False))
)

def rule_20_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == list_of_string_values.index(arg2))

        # Constraints for rule 20
        rule_20(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_20(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
