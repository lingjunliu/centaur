import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# Check for correct argument types when input is tensor and a single probability float is provided (Rule 15)

rule_15 = lambda s, v, n=False: (
    s.add(Not(And((v["arg1_ndim"] >= 0), (And(v["arg2_value"] >= 0, v["arg2_value"] <= 1)))) if n else
          And((v["arg1_ndim"] >= 0), (And(v["arg2_value"] >= 0, v["arg2_value"] <= 1))))
)

def rule_15_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, (float, np.floating)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = Real('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == arg2)

        # Constraints for rule 15
        rule_15(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_15(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
