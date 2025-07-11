import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if use_cudnn_on_gpu is false, then some other condition must hold - placeholder for more complex logic based on hardware or configuration, always true for now as we can't define specific hardware conditions (Rule 19)

rule_19 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == False, True, False)) if n else
          If(v["arg1_value"] == False, True, False))
)

def rule_19_func(arg1, solver=None, neg=False):
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

        # Constraints for rule 19
        rule_19(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_19(solver, {'arg1_value': arg1['value']}, neg)
