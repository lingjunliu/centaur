import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# When padding mode is replication and input is 3D tensor and padding value is tuple, then tuple length should be 2 (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 23, If(v["arg2_ndim"] == 3, v["arg3_length"] == 2, True), True)) if n else
          If(v["arg1_value"] == 23, If(v["arg2_ndim"] == 3, v["arg3_length"] == 2, True), True))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_torch.index(arg1))
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 48
        rule_48(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_length': arg3['length']}, neg)
