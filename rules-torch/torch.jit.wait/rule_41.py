import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# future is a list whose elements are the first few prime number (Rule 41)

rule_41 = lambda s, v, n=False: (
    s.add(Not(And(And(And(And(And((v["arg1_length"] <= 5), (If(v["arg1_length"] > 0, Select(v["arg1_values"], 0) == -100, True))), (If(v["arg1_length"] > 1, Select(v["arg1_values"], 1) == 90, True))), (If(v["arg1_length"] > 2, Select(v["arg1_values"], 2) == -1, True))), (If(v["arg1_length"] > 3, Select(v["arg1_values"], 3) == -97, True))), (If(v["arg1_length"] > 4, Select(v["arg1_values"], 4) == -70, True)))) if n else
          And(And(And(And(And((v["arg1_length"] <= 5), (If(v["arg1_length"] > 0, Select(v["arg1_values"], 0) == -100, True))), (If(v["arg1_length"] > 1, Select(v["arg1_values"], 1) == 90, True))), (If(v["arg1_length"] > 2, Select(v["arg1_values"], 2) == -1, True))), (If(v["arg1_length"] > 3, Select(v["arg1_values"], 3) == -97, True))), (If(v["arg1_length"] > 4, Select(v["arg1_values"], 4) == -70, True))))
)

def rule_41_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])

        # Constraints for rule 41
        rule_41(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_41(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values']}, neg)
