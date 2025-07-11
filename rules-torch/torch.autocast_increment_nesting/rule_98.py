import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If environment variable contains 'ij' and is autocast, increment must be 1 or 2 or 3. (Rule 98)

rule_98 = lambda s, v, n=False: (
    s.add(Not(If(And((Or(Or(v["arg1_value"] == 2, v["arg1_value"] == 3), v["arg1_value"] == 4)), v["arg2_value"] == True), Or(Or(v["arg3_value"] == 1, v["arg3_value"] == 2), v["arg3_value"] == 3), False)) if n else
          If(And((Or(Or(v["arg1_value"] == 2, v["arg1_value"] == 3), v["arg1_value"] == 4)), v["arg2_value"] == True), Or(Or(v["arg3_value"] == 1, v["arg3_value"] == 2), v["arg3_value"] == 3), False))
)

def rule_98_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 98
        rule_98(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_98(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
