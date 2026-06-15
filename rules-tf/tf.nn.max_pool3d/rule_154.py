import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# max_pool3d channels pooling is unsupported (Rule 154)

rule_154 = lambda s, v, n=False: (
    s.add(Not(If(v["arg2_value"] == 33, (If(v["arg1_length"] == 5, Select(v["arg1_values"], 4) == 1, True)), If(v["arg2_value"] == 34, (If(v["arg1_length"] == 5, Select(v["arg1_values"], 1) == 1, True)), True))) if n else
          If(v["arg2_value"] == 33, (If(v["arg1_length"] == 5, Select(v["arg1_values"], 4) == 1, True)), If(v["arg2_value"] == 34, (If(v["arg1_length"] == 5, Select(v["arg1_values"], 1) == 1, True)), True)))
)

def rule_154_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, list) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg1)):
            return False
        if not isinstance(arg2, str):
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
        solver.add(arg2_value == list_of_string_values_tf.index(arg2))

        # Constraints for rule 154
        rule_154(solver, {'arg1_values': arg1_values, 'arg1_length': arg1_length, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_154(solver, {'arg1_values': arg1['values'], 'arg1_length': arg1['length'], 'arg2_value': arg2['value']}, neg)
