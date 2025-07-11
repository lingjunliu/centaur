import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch.torch, np_dtype
from z3 import *

# bool variable must be equal to true or false based on a string comparison and dimension and min of tensor > 10  (Rule 111)

rule_111 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 7, v["arg1_value"] == True, If(v["arg3_ndim"] > 1, v["arg1_value"] == False, v["arg1_value"] == (Select(v["arg3_range"], 0) > 10)))) if n else
          If(v["arg2_value"] == 7, v["arg1_value"] == True, If(v["arg3_ndim"] > 1, v["arg1_value"] == False, v["arg1_value"] == (Select(v["arg3_range"], 0) > 10))))
)

def rule_111_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, str):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = String('arg2_value')
        arg3_ndim = Int('arg3_ndim')
        arg3_range = Array('arg3_range', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == list_of_string_values_torch.torch.index(arg2))
        solver.add(arg3_ndim == arg3.ndim)
        arg3_range = Store(arg3_range, 0, int(np.min(arg3)))
        arg3_range = Store(arg3_range, 1, int(np.max(arg3)))

        # Constraints for rule 111
        rule_111(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_ndim': arg3_ndim, 'arg3_range': arg3_range})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_111(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_ndim': arg3['ndim'], 'arg3_range': arg3['range']}, neg)
