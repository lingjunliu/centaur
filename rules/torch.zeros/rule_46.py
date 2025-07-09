import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If size is provided as a tuple and requires_grad is true, and out is provided, out cannot have float dtype (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_length"] > 0, v["arg2_value"] == True), v["arg3_dtype"] != 8, False)) if n else
          If(And(v["arg1_length"] > 0, v["arg2_value"] == True), v["arg3_dtype"] != 8, False))
)

def rule_46_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Bool('arg2_value')
        arg3_dtype = Int('arg3_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_dtype == list_of_available_dtypes.index(arg3.dtype))

        # Constraints for rule 46
        rule_46(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_dtype': arg3_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_dtype': arg3['dtype']}, neg)
