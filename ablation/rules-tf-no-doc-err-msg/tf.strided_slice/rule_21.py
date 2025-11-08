import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# The data type of begin, end, and strides should be integers (Rule 21)

rule_21 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And((And(Select(v["arg1_values"], i) > -2147483648, Select(v["arg1_values"], i) < 2147483647)), And([Implies(i < (v["arg2_length"] - 1 + 1), And((And(Select(v["arg2_values"], i) > -2147483648, Select(v["arg2_values"], i) < 2147483647)), And([Implies(i < (v["arg3_length"] - 1 + 1), (And(Select(v["arg3_values"], i) > -2147483648, Select(v["arg3_values"], i) < 2147483647))) for i in range(6)]))) for i in range(6)]))) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And((And(Select(v["arg1_values"], i) > -2147483648, Select(v["arg1_values"], i) < 2147483647)), And([Implies(i < (v["arg2_length"] - 1 + 1), And((And(Select(v["arg2_values"], i) > -2147483648, Select(v["arg2_values"], i) < 2147483647)), And([Implies(i < (v["arg3_length"] - 1 + 1), (And(Select(v["arg3_values"], i) > -2147483648, Select(v["arg3_values"], i) < 2147483647))) for i in range(6)]))) for i in range(6)]))) for i in range(6)]))
)

def rule_21_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg3)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())
        arg3_length = Int('arg3_length')
        arg3_values = Array('arg3_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])
        solver.add(arg3_length == len(arg3))
        for i in range(len(arg3)):
            arg3_values = Store(arg3_values, i, arg3[i])

        # Constraints for rule 21
        rule_21(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_values': arg2_values, 'arg2_length': arg2_length, 'arg3_values': arg3_values, 'arg3_length': arg3_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_21(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_values': arg2['values'], 'arg2_length': arg2['length'], 'arg3_values': arg3['values'], 'arg3_length': arg3['length']}, neg)
