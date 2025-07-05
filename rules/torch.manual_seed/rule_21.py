import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values, np_dtype
from z3 import *

# seed is valid if it's within 64 bit integer range or its remapped version is (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(Or(Or((And(v["arg1_value"] >= -9223372036854775808, v["arg1_value"] <= 9223372036854775807)), (And(And(v["arg1_value"] < 0, (v["arg1_value"] + 18446744073709551616) >= -9223372036854775808), (v["arg1_value"] + 18446744073709551616) <= 9223372036854775807))), (And(And(v["arg1_value"] >= 0, (v["arg1_value"]) >= -9223372036854775808), (v["arg1_value"]) <= 9223372036854775807)))) if n else
          Or(Or((And(v["arg1_value"] >= -9223372036854775808, v["arg1_value"] <= 9223372036854775807)), (And(And(v["arg1_value"] < 0, (v["arg1_value"] + 18446744073709551616) >= -9223372036854775808), (v["arg1_value"] + 18446744073709551616) <= 9223372036854775807))), (And(And(v["arg1_value"] >= 0, (v["arg1_value"]) >= -9223372036854775808), (v["arg1_value"]) <= 9223372036854775807))))
)

def rule_21_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, (int, np.integer)) and not isinstance(arg1, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')

        # Value assignments
        solver.add(arg1_value == int(arg1))

        # Constraints for rule 21
        rule_21(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_value': arg1['value']}, neg)
