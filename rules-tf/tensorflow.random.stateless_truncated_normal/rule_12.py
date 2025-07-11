import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_tf, np_dtype
from z3 import *

# Mean's dtype should be compatible with the output dtype (Rule 12)

rule_12 = lambda s, v, n=False: (
    s.add(Not(Or(Or(Or(Or(Or(v["arg1_value"] == v["arg2_value"], (And(v["arg1_value"] == 1, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 2, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 3, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 4, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 5, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10)))))) if n else
          Or(Or(Or(Or(Or(v["arg1_value"] == v["arg2_value"], (And(v["arg1_value"] == 1, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 2, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 3, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 4, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))), (And(v["arg1_value"] == 5, (Or(Or(Or(v["arg2_value"] == 7, v["arg2_value"] == 8), v["arg2_value"] == 9), v["arg2_value"] == 10))))))
)

def rule_12_func(arg1, arg2, solver=None, neg=False):
    arg1 = next(iter(arg1.values()))
    arg2 = next(iter(arg2.values()))

    # Invariant learning phase
    if not solver:
        if not (isinstance(arg1, torch.dtype) or isinstance(arg1, tf.dtypes.DType)):
            return False
        if not (isinstance(arg2, torch.dtype) or isinstance(arg2, tf.dtypes.DType)):
            return False

        # Variable declarations
        solver = Solver()
        arg1_value = Int('arg1_value')
        arg2_value = Int('arg2_value')

        # Value assignments
        solver.add(arg1_value == list_of_available_dtypes.index(np_dtype(arg1)))
        solver.add(arg2_value == list_of_available_dtypes.index(np_dtype(arg2)))

        # Constraints for rule 12
        rule_12(solver, {'arg1_value': arg1_value, 'arg2_value': arg2_value})
        return solver.check() == sat

    # Fuzz input generation phase
    else:
        rule_12(solver, {'arg1_value': arg1['value'], 'arg2_value': arg2['value']}, neg)
