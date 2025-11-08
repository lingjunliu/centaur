import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# A Bool condition should equal a tensor datatype equality or tuple length comparison to number. (Rule 151)

rule_151 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"], v["arg1_dtype"] == 7, v["arg3_length"] == 2)) if n else
          If(v["arg2_value"], v["arg1_dtype"] == 7, v["arg3_length"] == 2))
)

def rule_151_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_length = Int('arg3_length')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_length == len(arg3))

        # Constraints for rule 151
        rule_151(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_151(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_length': arg3['length']}, neg)
