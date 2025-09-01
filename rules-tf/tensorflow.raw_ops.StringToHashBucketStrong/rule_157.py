import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Check valid key list and that number of buckets is greater than zero to actually run the code (Rule 157)

rule_157 = lambda s, v, n=False: (
    s.add(Not(And(And((v["arg1_length"] == 2), (And([Implies(i < (v["arg1_length"] - 1 + 1), And((Select(v["arg1_values"], i) >= 0), (Select(v["arg1_values"], i) <= 2147483647))) for i in range(6)]))), (v["arg2_value"] > 0))) if n else
          And(And((v["arg1_length"] == 2), (And([Implies(i < (v["arg1_length"] - 1 + 1), And((Select(v["arg1_values"], i) >= 0), (Select(v["arg1_values"], i) <= 2147483647))) for i in range(6)]))), (v["arg2_value"] > 0)))
)

def rule_157_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, (int, np.integer)) and not isinstance(arg2, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_value == int(arg2))

        # Constraints for rule 157
        rule_157(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_157(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)
