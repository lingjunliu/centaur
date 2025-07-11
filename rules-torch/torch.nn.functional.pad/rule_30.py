import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If mode is replicate or reflect or circular, input dimension must be between 2 and 5 (Rule 30)

rule_30 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] != 10, And(v["arg1_ndim"] >= 2, v["arg1_ndim"] <= 5), False)) if n else
          If(v["arg2_value"] != 10, And(v["arg1_ndim"] >= 2, v["arg1_ndim"] <= 5), False))
)

def rule_30_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_value = String('arg2_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_value == list_of_string_values_torch.index(arg2))

        # Constraints for rule 30
        rule_30(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_30(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
