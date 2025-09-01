import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# If the size and stride are both non-empty and not equal in length then requires_grad and pin_memory must be false (Rule 46)

rule_46 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_length"] > 0, v["arg2_length"] > 0), v["arg1_length"] != v["arg2_length"]), And(v["arg3_value"] == False, v["arg4_value"] == False), True)) if n else
          If(And(And(v["arg1_length"] > 0, v["arg2_length"] > 0), v["arg1_length"] != v["arg2_length"]), And(v["arg3_value"] == False, v["arg4_value"] == False), True))
)

def rule_46_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_length = Int('arg2_length')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)

        # Constraints for rule 46
        rule_46(solver, {'arg1_length': arg1_length, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_46(solver, {'arg1_length': arg1['length'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
