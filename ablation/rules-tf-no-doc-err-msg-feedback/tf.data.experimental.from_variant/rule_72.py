import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# If string v_1 is not equal to either "valid" or "same", then tuple v_2 should have a length of 2 (Rule 72)

rule_72 = lambda s, v, n=False: (
    s.add(Not(If(And(v["arg1_value"] != 21, v["arg1_value"] != 22), v["arg2_length"] == 2, True)) if n else
          If(And(v["arg1_value"] != 21, v["arg1_value"] != 22), v["arg2_length"] == 2, True))
)

def rule_72_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False
        if not (isinstance(arg2, tuple) and all((isinstance(e, (int, np.integer)) and not isinstance(e, bool)) for e in arg2)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')
        arg2_length = Int('arg2_length')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))
        solver.add(arg2_length == len(arg2))

        # Constraints for rule 72
        rule_72(solver, {'arg1_value': arg1_value, 'arg2_length': arg2_length})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_72(solver, {'arg1_value': arg1['value'], 'arg2_length': arg2['length']}, neg)
