import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If value is int, dim entries can only be int32 or int64 (Rule 38)

rule_38 = lambda s, v, n=False: (
    s.add(Not(If(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), And([Implies(i < (v["arg1_length"] - 1 + 1), And((Select(v["arg1_values"], i) < 2147483647), (Select(v["arg1_values"], i) > -2147483648))) for i in range(6)]), True)) if n else
          If(Or(v["arg2_dtype"] == 3, v["arg2_dtype"] == 4), And([Implies(i < (v["arg1_length"] - 1 + 1), And((Select(v["arg1_values"], i) < 2147483647), (Select(v["arg1_values"], i) > -2147483648))) for i in range(6)]), True))
)

def rule_38_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, np.ndarray):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_dtype = Int('arg2_dtype')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_dtype == list_of_available_dtypes.index(arg2.dtype))

        # Constraints for rule 38
        rule_38(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_dtype': arg2_dtype})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_38(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_dtype': arg2['dtype']}, neg)
