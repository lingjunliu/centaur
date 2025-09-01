import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If images is of type uint16, uint32, uint64, complex64, complex128 then strides and rates must have the same size. (Rule 48)

rule_48 = lambda s, v, n=False: (
    s.add(Not(If(Or(Or(Or(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), v["arg2_length"] == v["arg3_length"], True)) if n else
          If(Or(Or(Or(Or(v["arg1_dtype"] == 9, v["arg1_dtype"] == 10), v["arg1_dtype"] == 11), v["arg1_dtype"] == 12), v["arg1_dtype"] == 13), v["arg2_length"] == v["arg3_length"], True))
)

def rule_48_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_length = Int('arg2_length')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 48
        rule_48(solver, {'arg1_dtype': arg1_dtype, 'arg2_length': arg2_length, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_48(solver, {'arg1_dtype': arg1['dtype'], 'arg2_length': arg2['length'], 'arg3_length': arg3['length']}, neg)
