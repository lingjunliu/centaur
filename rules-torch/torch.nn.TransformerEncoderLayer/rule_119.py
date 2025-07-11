import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# layer_norm_eps must be a small positive number, within (1e-06, 0.001 (Rule 119)

rule_119 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_value"] > 1e-6, v["arg1_value"] < 0.001), v["arg2_value"] % 8 == 0)) if n else
          And(And(v["arg1_value"] > 1e-6, v["arg1_value"] < 0.001), v["arg2_value"] % 8 == 0))
)

def rule_119_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, (float, np.floating)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Real('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 119
        rule_119(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_119(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
