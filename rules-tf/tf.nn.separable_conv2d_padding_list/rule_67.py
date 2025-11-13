import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If padding is valid or same, dilations should be 1,1,1,1 (Rule 67)

rule_67 = lambda s, v, n=False: (
    s.add(Not(If(Or((v["arg1_value"] == 21), (v["arg1_value"] == 22)), And(And(And((Select(v["arg2_values"], 0) == 1), (Select(v["arg2_values"], 1) == 1)), (Select(v["arg2_values"], 2) == 1)), (Select(v["arg2_values"], 3) == 1)), True)) if n else
          If(Or((v["arg1_value"] == 21), (v["arg1_value"] == 22)), And(And(And((Select(v["arg2_values"], 0) == 1), (Select(v["arg2_values"], 1) == 1)), (Select(v["arg2_values"], 2) == 1)), (Select(v["arg2_values"], 3) == 1)), True))
)

def rule_67_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 67
        rule_67(solver, {'arg1_value': arg1_value, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_67(solver, {'arg1_value': arg1['value'], 'arg2_values': arg2['values']}, neg)
