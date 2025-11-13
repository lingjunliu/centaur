import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# if there's more than 0 elements in inner_shape and the dtype is bigger than 5 or raggedRank is 0 then the first element of the list need to be smaller than 0 or greater than 0  (Rule 170)

rule_170 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg2_length"] > 0, (Or(v["arg3_value"] > 5, v["arg3_value"] == 0))), Or(Select(v["arg1_values"], 0) < 0, Select(v["arg1_values"], 0) > 0), True)) if n else
          If(And(v["arg2_length"] > 0, (Or(v["arg3_value"] > 5, v["arg3_value"] == 0))), Or(Select(v["arg1_values"], 0) < 0, Select(v["arg1_values"], 0) > 0), True))
)

def rule_170_func(arg1, arg2, arg3, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))
    arg3 = next(iter(arg3.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False
        if not (isinstance(arg3, (int, np.integer)) and not isinstance(arg3, bool)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_values = Array('arg1_values', IntSort(), IntSort())
        arg2_length = Int('arg2_length')
        arg3_value = Int('arg3_value')

        # Value assignments
        for i in range(len(arg1)):
            arg1_values = Store(arg1_values, i, arg1[i])
        solver.add(arg2_length == len(arg2))
        solver.add(arg3_value == int(arg3))

        # Constraints for rule 170
        rule_170(solver, {'arg1_values': arg1_values, 'arg2_length': arg2_length, 'arg3_value': arg3_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_170(solver, {'arg1_values': arg1['values'], 'arg2_length': arg2['length'], 'arg3_value': arg3['value']}, neg)
