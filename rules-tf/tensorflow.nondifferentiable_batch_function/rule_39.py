import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# When enable_large_batch_splitting is enabled and there are allowed batch sizes, the smallest allowed batch size should be a reasonable fraction of the largest allowed batch size, to make the splitting effective. (Rule 39)

rule_39 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] == True, v["arg2_length"] > 0), Select(v["arg2_values"], 0) > Select(v["arg2_values"], v["arg2_length"] - 1) / 8, True)) if n else
          If(And(v["arg1_value"] == True, v["arg2_length"] > 0), Select(v["arg2_values"], 0) > Select(v["arg2_values"], v["arg2_length"] - 1) / 8, True))
)

def rule_39_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, bool):
            return False
        if not (isinstance(arg2, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Bool('arg1_value')
        arg2_length = Int('arg2_length')
        arg2_values = Array('arg2_values', IntSort(), IntSort())

        # Value assignments
        solver.add(arg1_value == arg1)
        solver.add(arg2_length == len(arg2))
        for i in range(len(arg2)):
            arg2_values = Store(arg2_values, i, arg2[i])

        # Constraints for rule 39
        rule_39(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length, 'arg2_values': arg2_values})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_39(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length'], 'arg2_values': arg2['values']}, neg)
