import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If keepdim is False and dim contains all dims, the output is scalar (Rule 31)

rule_31 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"] == False, v["arg2_length"] == v["arg1_ndim"]), v["arg1_ndim"] == 0, True)) if n else
          If(And(v["arg3_value"] == False, v["arg2_length"] == v["arg1_ndim"]), v["arg1_ndim"] == 0, True))
)

def rule_31_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_ndim = Int('arg1_ndim')
        arg2_length = Int('arg2_length')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_ndim == arg1.ndim)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == arg3)

        # Constraints for rule 31
        rule_31(solver, {'arg1_ndim': arg1_ndim, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_31(solver, {'arg1_ndim': arg1['ndim'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
