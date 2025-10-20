import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If list consists of string, check types or expand composites must be false  (Rule 134)

rule_134 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_value"] == False, v["arg3_value"] == False), v["arg1_length"] > 0, True)) if n else
          If(Or(v["arg2_value"] == False, v["arg3_value"] == False), v["arg1_length"] > 0, True))
)

def rule_134_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all(isinstance(e, str) for e in arg1)):
            return False
        if not isinstance(arg2, bool):
            return False
        if not isinstance(arg3, bool):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg2_value = Bool('arg2_value')
        arg3_value = Bool('arg3_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        solver.add(arg2_value == arg2)
        solver.add(arg3_value == arg3)

        # Constraints for rule 134
        rule_134(solver, {'arg1_length': arg1_length, 'arg2_value': arg2_value, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_134(solver, {'arg1_length': arg1['length'], 'arg2_value': arg2['value'], 'arg3_value': arg3['value']}, neg)
