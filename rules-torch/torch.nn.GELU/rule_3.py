import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# approximate argument must be either none or tanh - using != (Rule 3)

rule_3 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_value"] != 11, v["arg1_value"] != 13), v["arg1_value"] != 15), v["arg1_value"] != 16), v["arg1_value"] != 18), v["arg1_value"] != 19), v["arg1_value"] != 20), v["arg1_value"] != 21), v["arg1_value"] != 22), v["arg1_value"] != 23), v["arg1_value"] != 24), v["arg1_value"] != 25), v["arg1_value"] != 26), v["arg1_value"] != 27), v["arg1_value"] != 28), v["arg1_value"] != 29)) if n else
          And(And(And(And(And(And(And(And(And(And(And(And(And(And(And(v["arg1_value"] != 11, v["arg1_value"] != 13), v["arg1_value"] != 15), v["arg1_value"] != 16), v["arg1_value"] != 18), v["arg1_value"] != 19), v["arg1_value"] != 20), v["arg1_value"] != 21), v["arg1_value"] != 22), v["arg1_value"] != 23), v["arg1_value"] != 24), v["arg1_value"] != 25), v["arg1_value"] != 26), v["arg1_value"] != 27), v["arg1_value"] != 28), v["arg1_value"] != 29))
)

def rule_3_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))

        # Constraints for rule 3
        rule_3(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_3(solver, {'arg1_value': arg1['value']}, neg)
