import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check that if v_1 then the length of v_2 is equal to v_3 else the length of v_2 is equal to v_4 (Rule 42)

rule_42 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"], v["arg2_length"] == v["arg3_value"], v["arg2_length"] == v["arg4_value"])) if n else
          If(v["arg1_value"], v["arg2_length"] == v["arg3_value"], v["arg2_length"] == v["arg4_value"]))
)

def rule_42_func(arg1, arg2, arg3, arg4, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False
        if not (isinstance(arg4, (int, np.integer)) and not isinstance(arg4, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg3_value = Int('arg3_value')
        arg4_value = Int('arg4_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == int(arg3))
        solver.add(arg4_value == int(arg4))

        # Constraints for rule 42
        rule_42(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg3_value': arg3_value, 'arg4_value': arg4_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_42(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value']}, neg)
