import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# executing_eagerly result determined by context flags (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(If(v["arg6_value"], v["arg1_value"] == False, If(v["arg5_value"], v["arg1_value"] == False, If(v["arg2_value"], If(Or(v["arg3_value"], v["arg4_value"]), v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == True)))) if n else
          If(v["arg6_value"], v["arg1_value"] == False, If(v["arg5_value"], v["arg1_value"] == False, If(v["arg2_value"], If(Or(v["arg3_value"], v["arg4_value"]), v["arg1_value"] == True, v["arg1_value"] == False), v["arg1_value"] == True))))
)

def rule_10_func(arg1, arg2, arg3, arg4, arg5, arg6, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))
    arg4 = next(iter(arg4.values()))
    arg5 = next(iter(arg5.values()))
    arg6 = next(iter(arg6.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False
        if not isinstance(arg4, bool):
            return False
        if not isinstance(arg5, bool):
            return False
        if not isinstance(arg6, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')
        arg4_value = Bool('arg4_value')
        arg5_value = Bool('arg5_value')
        arg6_value = Bool('arg6_value')

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)
        solver.add(arg4_value == arg4)
        solver.add(arg5_value == arg5)
        solver.add(arg6_value == arg6)

        # Constraints for rule 10
        rule_10(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value, 'arg3_value': arg3_value, 'arg4_value': arg4_value, 'arg5_value': arg5_value, 'arg6_value': arg6_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value'], 'arg4_value': arg4['value'], 'arg5_value': arg5['value'], 'arg6_value': arg6['value']}, neg)
