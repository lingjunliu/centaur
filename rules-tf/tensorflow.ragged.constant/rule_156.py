import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If the first pylist item is not 1 or 0 and row splits can't have zero elements. (Rule 156)

rule_156 = lambda s, v, n=False: (
    s.add(Not(If(And(And(v["arg1_length"] > 0, Select(v["arg1_values"], 0) != 1), Select(v["arg1_values"], 0) != 0), v["arg2_value"] > 0, True)) if n else
          If(And(And(v["arg1_length"] > 0, Select(v["arg1_values"], 0) != 1), Select(v["arg1_values"], 0) != 0), v["arg2_value"] > 0, True))
)

def rule_156_func(arg1, arg2, solver=None, neg=False):
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

        # Constraints for rule 156
        rule_156(solver, {'arg1_length': arg1_length, 'arg1_values': arg1_values, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_156(solver, {'arg1_length': arg1['length'], 'arg1_values': arg1['values'], 'arg2_value': arg2['value']}, neg)
