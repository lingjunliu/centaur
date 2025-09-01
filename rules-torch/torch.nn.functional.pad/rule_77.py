import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# Combined rule for mode and replicate value (Rule 77)

rule_77 = lambda s, v, n=False: (
    s.add(Not(And((Or(Or(Or(v["arg1_value"] == 21, v["arg1_value"] == 22), v["arg1_value"] == 23), v["arg1_value"] == 24)), (If(v["arg1_value"] == 23, v["arg2_value"] == 0, True)))) if n else
          And((Or(Or(Or(v["arg1_value"] == 21, v["arg1_value"] == 22), v["arg1_value"] == 23), v["arg1_value"] == 24)), (If(v["arg1_value"] == 23, v["arg2_value"] == 0, True))))
)

def rule_77_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_value == arg2)

        # Constraints for rule 77
        rule_77(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_77(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
