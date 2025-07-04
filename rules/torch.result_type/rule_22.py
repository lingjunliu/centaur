import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# If one input is int and the other is int and v1>=v2, the result is int (with largest range (Rule 22)

rule_22 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] >= v["arg2_value"], Or(Or(Or(Or((And(And(And(v["arg1_value"] >= -128, v["arg1_value"] <= 127), v["arg2_value"] >= -128), v["arg2_value"] <= 127)), (And(And(And(v["arg1_value"] >= -32768, v["arg1_value"] <= 32767), v["arg2_value"] >= -32768), v["arg2_value"] <= 32767))), (And(And(And(v["arg1_value"] >= -2147483648, v["arg1_value"] <= 2147483647), v["arg2_value"] >= -2147483648), v["arg2_value"] <= 2147483647))), (And(And(And(v["arg1_value"] >= -9223372036854775808, v["arg1_value"] <= 9223372036854775807), v["arg2_value"] >= -9223372036854775808), v["arg2_value"] <= 9223372036854775807))), (And(And(And(v["arg1_value"] >= 0, v["arg1_value"] <= 255), v["arg2_value"] >= 0), v["arg2_value"] <= 255))), False)) if n else
          If(v["arg1_value"] >= v["arg2_value"], Or(Or(Or(Or((And(And(And(v["arg1_value"] >= -128, v["arg1_value"] <= 127), v["arg2_value"] >= -128), v["arg2_value"] <= 127)), (And(And(And(v["arg1_value"] >= -32768, v["arg1_value"] <= 32767), v["arg2_value"] >= -32768), v["arg2_value"] <= 32767))), (And(And(And(v["arg1_value"] >= -2147483648, v["arg1_value"] <= 2147483647), v["arg2_value"] >= -2147483648), v["arg2_value"] <= 2147483647))), (And(And(And(v["arg1_value"] >= -9223372036854775808, v["arg1_value"] <= 9223372036854775807), v["arg2_value"] >= -9223372036854775808), v["arg2_value"] <= 9223372036854775807))), (And(And(And(v["arg1_value"] >= 0, v["arg1_value"] <= 255), v["arg2_value"] >= 0), v["arg2_value"] <= 255))), False))
)

def rule_22_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 22
        rule_22(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_22(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
