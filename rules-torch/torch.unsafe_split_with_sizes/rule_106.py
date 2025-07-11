import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# The integral type is correct IF certain flag conditions on the data exist (Rule 106)

rule_106 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg3_value"], v["arg4_length"] > 0), (And(And(v["arg2_value"], 1 <= v["arg1_dtype"]), v["arg1_dtype"] <= 5)), False)) if n else
          If(And(v["arg3_value"], v["arg4_length"] > 0), (And(And(v["arg2_value"], 1 <= v["arg1_dtype"]), v["arg1_dtype"] <= 5)), False))
)

def rule_106_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, np.ndarray):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not (isinstance(arg4, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg4)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_dtype = Int('arg1_dtype')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_length = Int('arg4_length')

        # Value assignments
        solver.add(arg1_dtype == list_of_available_dtypes.index(arg1.dtype))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_length == len(arg4))

        # Constraints for rule 106
        rule_106(solver, {'arg1_dtype': arg1_dtype, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_length': arg4_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_106(solver, {'arg1_dtype': arg1['dtype'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_length': arg4['length']}, neg)
