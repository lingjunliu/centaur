import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# String v1 should be equal to one of the interpolation modes if the tensor's dimension is 4 (Rule 55)

rule_55 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_ndim"] == 4, (Or(Or(Or(Or(Or(v["arg2_value"] == 25, v["arg2_value"] == 20), v["arg2_value"] == 26), v["arg2_value"] == 27), v["arg2_value"] == 28), v["arg2_value"] == 29)), True)) if n else
          If(v["arg1_ndim"] == 4, (Or(Or(Or(Or(Or(v["arg2_value"] == 25, v["arg2_value"] == 20), v["arg2_value"] == 26), v["arg2_value"] == 27), v["arg2_value"] == 28), v["arg2_value"] == 29)), True))
)

def rule_55_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 55
        rule_55(solver, {'arg1_ndim': arg1_ndim, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_55(solver, {'arg1_ndim': arg1['ndim'], 'arg2_value': arg2['value']}, neg)
