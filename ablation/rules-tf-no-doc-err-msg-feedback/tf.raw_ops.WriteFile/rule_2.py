import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# filename is a valid file path if it contains ".txt" or ".csv" (Rule 2)

rule_2 = lambda s, v, n=False: (
    s.add(Not(If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] == 9, True, If(v["arg1_value"] == 10, True, If(v["arg1_value"] == 11, True, If(v["arg1_value"] == 12, True, If(v["arg1_value"] == 13, True, If(v["arg1_value"] == 14, True, If(v["arg1_value"] == 15, True, If(v["arg1_value"] == 16, True, If(v["arg1_value"] == 17, True, If(v["arg1_value"] == 18, True, If(v["arg1_value"] == 19, True, If(v["arg1_value"] == 20, True, If(v["arg1_value"] == 21, True, If(v["arg1_value"] == 22, True, If(v["arg1_value"] == 23, True, If(v["arg1_value"] == 24, True, If(v["arg1_value"] == 25, True, True))))))))))))))))))))) if n else
          If(v["arg1_value"] == 6, True, If(v["arg1_value"] == 7, True, If(v["arg1_value"] == 8, True, If(v["arg1_value"] == 9, True, If(v["arg1_value"] == 10, True, If(v["arg1_value"] == 11, True, If(v["arg1_value"] == 12, True, If(v["arg1_value"] == 13, True, If(v["arg1_value"] == 14, True, If(v["arg1_value"] == 15, True, If(v["arg1_value"] == 16, True, If(v["arg1_value"] == 17, True, If(v["arg1_value"] == 18, True, If(v["arg1_value"] == 19, True, If(v["arg1_value"] == 20, True, If(v["arg1_value"] == 21, True, If(v["arg1_value"] == 22, True, If(v["arg1_value"] == 23, True, If(v["arg1_value"] == 24, True, If(v["arg1_value"] == 25, True, True)))))))))))))))))))))
)

def rule_2_func(arg1, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))

    # Invariant learning phase
    if not solver:
        if not isinstance(arg1, str):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = String('arg1_value')

        # Value assignments
        solver.add(arg1_value == list_of_string_values_tf.index(arg1))

        # Constraints for rule 2
        rule_2(solver, {'arg1_value': arg1_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_2(solver, {'arg1_value': arg1['value']}, neg)
