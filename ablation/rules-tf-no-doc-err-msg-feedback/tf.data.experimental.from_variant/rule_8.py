import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If v_1 is true then dimensions of tensor v_2 must be equal to integer v_3 (Rule 8)

rule_8 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], v["arg2_ndim"] == v["arg3_value"], True)) if n else
          If(v["arg1_value"], v["arg2_ndim"] == v["arg3_value"], True))
)

def rule_8_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, np.ndarray):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_ndim = Int('arg2_ndim')
        arg3_value = Int('arg3_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_ndim == arg2.ndim)
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 8
        rule_8(solver, {'arg1_value': arg1_value, 'arg2_ndim': arg2_ndim, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_8(solver, {'arg1_value': arg1['value'], 'arg2_ndim': arg2['ndim'], 'arg3_value': arg3['value']}, neg)
