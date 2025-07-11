import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# torch.get_num_threads accepts no parameters, any string v_1 will match itself (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(v["arg1_value"] == v["arg1_value"]) if n else
          v["arg1_value"] == v["arg1_value"])
)

def rule_22_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value']}, neg)
