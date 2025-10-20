import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_torch, np_dtype
from z3 import *

# pad should be smaller than or equal to half of kernel size (Rule 10)

rule_10 = lambda s, v, n=False: (
    s.add(Not(And(And(v["arg1_length"] == 3, v["arg2_length"] == 3), And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) >= 0, Select(v["arg2_values"], i) > 0), Select(v["arg1_values"], i) <= Select(v["arg2_values"], i) / 2)) for i in range(6)]))) if n else
          And(And(v["arg1_length"] == 3, v["arg2_length"] == 3), And([Implies(i < (v["arg1_length"] - 1 + 1), And(And(Select(v["arg1_values"], i) >= 0, Select(v["arg2_values"], i) > 0), Select(v["arg1_values"], i) <= Select(v["arg2_values"], i) / 2)) for i in range(6)])))
)

def rule_10_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 10
        rule_10(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_10(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
