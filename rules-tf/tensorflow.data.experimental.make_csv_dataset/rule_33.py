import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If select_columns are indices, each index must be within range. (Rule 33)

rule_33 = lambda s, v, n=False: (
    s.add(Not(And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < v["arg2_length"])) for i in range(6)])) if n else
          And([Implies(i < (v["arg1_length"] - 1 + 1), And(Select(v["arg1_values"], i) >= 0, Select(v["arg1_values"], i) < v["arg2_length"])) for i in range(6)]))
)

def rule_33_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, list) and all(isinstance(e, str) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_length = Int('arg1_length')
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_length == len(arg1))
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 33
        rule_33(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_33(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_length': arg2['length']}, neg)
